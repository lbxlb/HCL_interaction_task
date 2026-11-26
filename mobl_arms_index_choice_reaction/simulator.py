from pathlib import Path

import mujoco
import mujoco.viewer  
import numpy as np
import yaml
import time
import os

# 兼容所有MuJoCo版本+远程环境
os.environ['MUJOCO_GL'] = 'egl'
os.environ['MJPYTHON_FRAMEWORK'] = 'none'
os.environ['MESA_GL_VERSION_OVERRIDE'] = '3.3'

class IndexSimulator:
    def __init__(self, config_path, model_path):
        with open(config_path, encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        self.model = mujoco.MjModel.from_xml_path(str(model_path))
        # 配置中的 dt 覆盖模型步长；未配置时保留 XML 的设置。
        dt = self.config.get('dt', self.model.opt.timestep)
        if isinstance(dt, bool) or not isinstance(dt, (int, float)):
            raise ValueError("dt 必须是有限的正数，单位为秒")
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError("dt 必须是有限的正数，单位为秒")
        self.model.opt.timestep = dt
        self.data = mujoco.MjData(self.model)
        
        # 1. 模型关键信息+关节range（稳定核心）
        print("\n📌 模型核心信息：")
        print(f"关节总数：{self.model.njnt} | 执行器数：{self.model.nu} | qpos长度：{len(self.data.qpos)}")
        print(f"物理步长：{self.model.opt.timestep:g} 秒/步")

        # 在__init__里新增：打印所有geom
        print("\n📌 所有geom名称：")
        for i in range(self.model.ngeom):
            geom_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, i)
            print(f"geom{i}：{geom_name}")
        
        self.key_joints = {
            "shoulder_rot": {
                "name": "shoulder_rot (肩关节旋转)",
                "speed": 0.5,  # 演示角速度幅值，rad/s
                "direction": 1,
            },
            "elbow_flexion": {
                "name": "elbow_flexion (肘关节弯曲)",
                "speed": 0.8,
                "direction": 1,
            },
            "pro_sup": {
                "name": "pro_sup (前臂旋前旋后)",
                "speed": 0.4,
                "direction": 1,
            }
        }

        speeds = self.config.get('demo_joint_speeds', {})
        if not isinstance(speeds, dict) or speeds.keys() - self.key_joints.keys():
            raise ValueError("demo_joint_speeds 必须是仅包含三个关键关节名称的映射")
        # joint_id 索引关节属性；qpos_adr 索引位置，两者不能混用。
        for joint_name, info in self.key_joints.items():
            speed = speeds.get(joint_name, info['speed'])
            if (isinstance(speed, bool) or not isinstance(speed, (int, float))
                    or not np.isfinite(speed) or speed <= 0):
                raise ValueError(f"{joint_name} 的演示速度必须是有限正数，单位 rad/s")
            info['speed'] = speed
            joint_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name
            )
            if joint_id == -1:
                raise ValueError(f"模型缺少必需关节：{joint_name}")
            if (self.model.jnt_type[joint_id] != mujoco.mjtJoint.mjJNT_HINGE
                    or not self.model.jnt_limited[joint_id]):
                raise ValueError(f"关节 {joint_name} 必须是有活动范围限制的 hinge 关节")
            info['joint_id'] = joint_id
            info['qpos_adr'] = int(self.model.jnt_qposadr[joint_id])
            info['range'] = tuple(self.model.jnt_range[joint_id])

        # 打印关节初始信息
        print("\n📌 关键关节（带range限制）：")
        for info in self.key_joints.values():
            init_val = (info['range'][0] + info['range'][1]) / 2
            print(f"关节{info['joint_id']}：{info['name']} | qpos地址：{info['qpos_adr']} | range：{info['range']} | 重置值：{init_val:.2f} rad | 演示速度：{info['speed']:g} rad/s")

        # 2. 仿真状态（极简，避免段错误）
        self.is_running = True
        self.viewer = None
        self.print_interval = 50
        self.current_step = 0
        self.button_touched = {"button-0":False, "button-1":False, "button-2":False, "button-3":False}
        self.finger_geom_name = "hand_2distph"
        # 独立参考状态只用于轨迹检查，尚未作为控制器输入。
        self.reference_qpos = self.data.qpos.copy()
        self.reference_directions = {name: 1 for name in self.key_joints}

    def reset(self):
        """重置关节到range中间值"""
        mujoco.mj_resetData(self.model, self.data)
        for info in self.key_joints.values():
            init_val = (info['range'][0] + info['range'][1]) / 2
            self.data.qpos[info['qpos_adr']] = init_val
            info['direction'] = 1  # 重置为正方向
        mujoco.mj_forward(self.model, self.data)
        self.reference_qpos = self.data.qpos.copy()
        self.reference_directions = {name: 1 for name in self.key_joints}
        self.current_step = 0
        self.is_running = True
        return self.data.qpos.copy()

    def _advance_position(self, position, direction, info):
        """一个演示增量；返回有限位的位置和下一步方向。"""
        value = position + direction * info['speed'] * self.model.opt.timestep
        lo, hi = info['range']
        if value >= hi:
            direction = -1
        elif value <= lo:
            direction = 1
        return float(np.clip(value, lo, hi)), direction

    def step(self):
        """窗口关闭时停止；否则推进一个物理步并同步画面。"""
        if self.viewer is not None and not self.viewer.is_running():
            self.is_running = False
        if not self.is_running:
            return self.data.qpos.copy()

        # 1. 演示位置增量 = 角速度 × dt；实际响应仍会受物理积分影响。
        for name, info in self.key_joints.items():
            adr = info['qpos_adr']
            self.reference_qpos[adr], self.reference_directions[name] = self._advance_position(
                self.reference_qpos[adr], self.reference_directions[name], info)
            # 保留原有实际运动逻辑；不把参考位置强行覆盖到动力学状态。
            self.data.qpos[adr], info['direction'] = self._advance_position(
                self.data.qpos[adr], info['direction'], info)

        # 2. 推进仿真
        mujoco.mj_step(self.model, self.data)
        # 积分已更新 qpos；刷新位置和碰撞，让画面与任务判定使用当前姿态。
        # mj_forward 不推进仿真时间。
        mujoco.mj_forward(self.model, self.data)

        # 3. 定期打印（减少刷屏）
        if self.current_step % self.print_interval == 0:
            print(f"\n📌 step{self.current_step} 关节角度（直观度数）：")
            for info in self.key_joints.values():
                rad = self.data.qpos[info['qpos_adr']]
                deg = rad * 57.3  # 弧度转角度
                print(f"   {info['name']}：{rad:.2f} rad ≈ {deg:.0f}°")

        # 4. 每步只在这里同步画面；同步期间窗口也可能关闭。
        if self.viewer is not None:
            try:
                if self.viewer.is_running():
                    self.viewer.sync()
                if self.viewer.is_running():
                    time.sleep(0.03)  # 仅影响现实播放速度
                else:
                    self.is_running = False
            except Exception as e:
                print(f"⚠️ 渲染同步失败，停止仿真：{e}")
                self.is_running = False

        self.current_step += 1
        return self.data.qpos.copy()

    def run_simulation(self):
        """运行动作演示，关闭窗口或按 Ctrl+C 退出。"""
        print("\n✅ 仿真启动成功！")
        print("👉 手臂会摆动，关闭窗口或按【Ctrl+C】终止仿真")
        self.reset()

        # 启动可视化（最大容错）
        try:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            # 优化视角
            self.viewer.cam.azimuth = 135
            self.viewer.cam.elevation = -15
            self.viewer.cam.distance = 0.6
            self.viewer.cam.lookat = [0.45, -0.15, 0.8]
            print("✅ 可视化窗口启动成功！能看到手臂慢速摆动")
        except Exception as e:
            print(f"⚠️ 可视化启动失败（不影响动作）：{e}")
            print("📢 无窗口模式：终端打印关节角度，确认动作真实发生！")

        # 退出或运行异常时均清理窗口资源。
        try:
            while self.is_running:
                self.step()
        except KeyboardInterrupt:
            print("\n\n⚠️ 检测到Ctrl+C，正在优雅退出仿真...")
            self.is_running = False

        finally:
            self.close()
        print(f"\n✅ 仿真正常结束！共运行{self.current_step}步")
        print("📊 最终关节角度：")
        for info in self.key_joints.values():
            rad = self.data.qpos[info['qpos_adr']]
            deg = rad * 57.3
            print(f"   {info['name']}：{rad:.2f} rad ≈ {deg:.0f}°")

    def close(self):
        """停止仿真并释放窗口；允许重复调用。"""
        self.is_running = False
        if self.viewer is not None:
            try:
                self.viewer.close()
            except Exception as e:
                print(f"⚠️ 窗口清理失败：{e}")
            finally:
                self.viewer = None

if __name__ == "__main__":
    # 独立演示也从脚本所在目录查找默认资源。
    BASE_DIR = Path(__file__).resolve().parent
    CONFIG_PATH = BASE_DIR / "config.yaml"
    MODEL_PATH = BASE_DIR / "simulation.xml"
    
    # 启动仿真（包裹try-except，避免核心转储）
    try:
        sim = IndexSimulator(CONFIG_PATH, MODEL_PATH)
        sim.run_simulation()
    except Exception as e:
        print(f"\n❌ 仿真启动异常：{e}")
        print("💡 请检查模型文件路径/配置文件是否正确")
