import mujoco
import numpy as np

class ChoicePanelTask:
    def __init__(self, config, sim):
        self.config = config
        self.sim = sim  # 关联你的仿真器实例
        # 从配置中读取任务参数（后续在config.yaml中配置）
        self.target_button = config.get("target_button", 0)  # 默认目标按钮0
        if type(self.target_button) is not int or self.target_button not in range(4):
            raise ValueError("target_button 必须是 0、1、2 或 3 的整数")
        self.screen_geom_id = mujoco.mj_name2id(
            sim.model, mujoco.mjtObj.mjOBJ_GEOM, "screen"
        )
        self.target_geom_id = mujoco.mj_name2id(
            sim.model, mujoco.mjtObj.mjOBJ_GEOM, f"button-{self.target_button}"
        )
        if self.screen_geom_id == -1 or self.target_geom_id == -1:
            raise ValueError(f"模型必须包含 screen 和 button-{self.target_button} 几何体")
        self.button_reward = config.get("button_reward", 10.0)  # 选对奖励
        self.effort_cost = config.get("effort_cost", 0.01)  # 动作成本系数
        self.timeout_steps = config.get("timeout", 300)  # 超时步数
        
        self.step_count = 0  # 记录当前任务步数
        self.done = False    # 任务是否完成
        self.reward = 0.0    # 当前奖励值
        self.total_reward = 0.0  # 本轮累计奖励
        self.outcome = "running"
        self.start_time = float(self.sim.data.time)
        self.elapsed_time = 0.0
        self.reaction_time = None  # 仅成功时有值，单位为仿真秒

    def reset(self):
        """重置任务状态（每次仿真/任务重新开始时调用）"""
        # 提示颜色直接读取目标按钮，和成功判定共用同一个目标。
        self.sim.model.geom_rgba[self.screen_geom_id] = self.sim.model.geom_rgba[self.target_geom_id]
        self.step_count = 0
        self.done = False
        self.reward = 0.0
        self.total_reward = 0.0
        self.outcome = "running"
        self.start_time = float(self.sim.data.time)
        self.elapsed_time = 0.0
        self.reaction_time = None
        timeout_seconds = self.timeout_steps * self.sim.model.opt.timestep
        print(f"\n🔄 任务已重置！目标按钮：button-{self.target_button}，超时步数：{self.timeout_steps}（{timeout_seconds:g} 秒仿真时间）")
        return self._status()

    def _status(self):
        """返回快照；reward 是当前步奖励，total_reward 是本轮累计值。"""
        return {
            "reward": self.reward,
            "done": self.done,
            "target_button": self.target_button,
            "outcome": self.outcome,
            "step_count": self.step_count,
            "elapsed_time": self.elapsed_time,
            "reaction_time": self.reaction_time,
            "total_reward": self.total_reward,
        }

    def update(self):
        """每一步仿真都调用，更新任务状态（核心逻辑）"""
        if self.done:  # 任务已完成，直接返回状态
            return self._status()
        
        self.step_count += 1
        self.reward = 0.0
        self.elapsed_time = float(self.sim.data.time) - self.start_time

        # 1. 检测：指尖是否碰到目标按钮
        is_touch_target = self._check_button_contact(self.target_button)
        if is_touch_target:
            self.reward += self.button_reward  # 选对按钮，添加奖励
            self.done = True
            self.outcome = "success"
            self.reaction_time = self.elapsed_time

        # 2. 动作成本扣分（兼容执行器为0的情况）
        ctrl_effort = np.sum(np.abs(self.sim.data.ctrl)) if self.sim.model.nu > 0 else 0.0
        effort_penalty = self.effort_cost * ctrl_effort
        self.reward -= effort_penalty
        if self.sim.model.nu == 0 and self.step_count % 50 == 0:
            print(f"\n💡 提示：当前模型无执行器，动作成本扣分暂不生效（后续添加执行器后自动启用）")

        # 3. 截止步包含在有效选择范围内：当步成功后不再判超时。
        if not self.done and self.step_count >= self.timeout_steps:
            self.done = True
            self.outcome = "timeout"
            self.reward -= 5.0  # 超时额外扣分

        self.total_reward += self.reward
        if self.done:
            result = "成功" if self.outcome == "success" else "超时"
            # 记录本轮结果到 artifacts/choice_reaction_logs/episode_results.json
            try:
                from pathlib import Path
                import json
                log_dir = Path(__file__).resolve().parent.parent / 'artifacts' / 'choice_reaction_logs'
                log_dir.mkdir(parents=True, exist_ok=True)
                log_file = log_dir / 'episode_results.json'
                history = []
                if log_file.exists():
                    try:
                        history = json.loads(log_file.read_text(encoding='utf-8'))
                    except Exception:
                        history = []
                episode_record = dict(
                    target_button=self.target_button,
                    outcome=self.outcome,
                    step_count=self.step_count,
                    elapsed_time=self.elapsed_time,
                    reaction_time=self.reaction_time,
                    total_reward=self.total_reward
                )
                history.append(episode_record)
                log_file.write_text(json.dumps(history, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
            except Exception as e:
                print(f"⚠️ 记录实验日志失败: {e}")
            reaction = f"{self.reaction_time:.6f} 秒" if self.reaction_time is not None else "无（未成功）"
            print(f"\n本轮结果：{result} | 目标：button-{self.target_button} | 步数：{self.step_count}"
                  f" | 耗时：{self.elapsed_time:.6f} 秒（仿真时间） | 反应时间：{reaction}"
                  f" | 本步奖励：{self.reward:.2f} | 累计奖励：{self.total_reward:.2f}")

        # 4. 定期打印任务状态（每50步）
        if self.step_count % 50 == 0 and not self.done:
            print(f"\n💰 step{self.step_count} 任务状态：")
            print(f"   是否碰到目标按钮：{'是' if is_touch_target else '否'}")
            print(f"   当前奖励：{self.reward:.2f}")
            print(f"   剩余步数：{self.timeout_steps - self.step_count}")

        return self._status()

    def _check_button_contact(self, button_id):
        """检测指尖（hand_2distph）是否碰到指定按钮（button-x）"""
        finger_geom_name = self.sim.finger_geom_name  # 从仿真器获取指尖geom名称
        target_button_name = f"button-{button_id}"    # 目标按钮geom名称

        # 遍历所有碰撞对，判断是否包含「指尖」和「目标按钮」
        for i in range(self.sim.data.ncon):
            contact = self.sim.data.contact[i]
            # 获取两个碰撞geom的名称
            geom1_id = contact.geom1
            geom2_id = contact.geom2
            geom1_name = mujoco.mj_id2name(self.sim.model, mujoco.mjtObj.mjOBJ_GEOM, geom1_id)
            geom2_name = mujoco.mj_id2name(self.sim.model, mujoco.mjtObj.mjOBJ_GEOM, geom2_id)

            # 判断是否是「指尖-目标按钮」的碰撞
            if (finger_geom_name in [geom1_name, geom2_name]) and (target_button_name in [geom1_name, geom2_name]):
                return True
        return False

