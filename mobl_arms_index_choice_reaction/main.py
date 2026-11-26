from pathlib import Path
import argparse

import mujoco
from simulator import IndexSimulator
from task import ChoicePanelTask  # 新增：导入任务类
import yaml  # 新增：加载配置文件

def main(argv=None):
    parser = argparse.ArgumentParser(description="面板选择任务")
    parser.add_argument('--headless', action='store_true', help="不启动可视化窗口；需指定 --max-steps")
    parser.add_argument('--max-steps', type=int, help="本次运行的物理总步数上限，跨任务重置累计")
    args = parser.parse_args(argv)
    if args.max_steps is not None and args.max_steps <= 0:
        parser.error('--max-steps 必须是正整数')
    if args.headless and args.max_steps is None:
        parser.error('--headless 需配合 --max-steps 使用')

    # 默认资源相对于入口脚本定位，不依赖终端当前目录。
    base_dir = Path(__file__).resolve().parent
    config_path = base_dir / "config.yaml"
    model_path = base_dir / "simulation.xml"

    # 新增：加载配置文件（给任务传参）
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # 2. 初始化仿真器+任务实例
    sim = IndexSimulator(config_path, model_path)
    task = ChoicePanelTask(config, sim)  # 关联配置和仿真器

    # 3. 重置仿真和任务
    sim.reset()
    task.reset()

    total_steps = 0
    total_sim_time = 0.0
    # 4. 仿真循环（整合任务逻辑，替代sim.run_simulation()）
    try:
        if args.headless:
            print(f"无窗口运行，上限 {args.max_steps} 个物理步")
        else:
            sim.viewer = mujoco.viewer.launch_passive(sim.model, sim.data)
            # 优化视角
            sim.viewer.cam.azimuth = 135
            sim.viewer.cam.elevation = -15
            sim.viewer.cam.distance = 0.6
            sim.viewer.cam.lookat = [0.45, -0.15, 0.8]
            print("✅ 可视化窗口启动成功！能看到手臂慢速摆动")

        # 循环执行仿真和任务更新
        while sim.is_running and (args.max_steps is None or total_steps < args.max_steps):
            # 仿真步进（模型运动）
            previous_step = sim.current_step
            sim.step()
            advanced_steps = sim.current_step - previous_step
            total_steps += advanced_steps
            total_sim_time += advanced_steps * sim.model.opt.timestep
            if not sim.is_running:
                break  # 窗口已关闭，不再更新任务或重置新一轮。
            # 任务状态更新（判断成败、计算奖励）
            task_status = task.update()
            if args.max_steps is not None and total_steps >= args.max_steps:
                print(f"已达到总步数上限：{args.max_steps}")
                break

            # 如果任务完成（成功/超时），重置仿真和任务
            if task_status["done"]:
                print("\n🔄 准备开始新一轮任务...")
                sim.reset()
                task.reset()

    except KeyboardInterrupt:
        print("\n\n⚠️ 检测到Ctrl+C，正在优雅退出仿真...")
        sim.is_running = False
    finally:
        # 关闭资源
        sim.close()
        print(f"\n运行结束：总步数 {total_steps}，累计仿真时间 {total_sim_time:.6f} 秒")

if __name__ == "__main__":
    main()
