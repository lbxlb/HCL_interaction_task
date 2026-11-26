# 实验复现指南

从仓库根目录运行以下命令。面板仿真使用 Python 3.10、MuJoCo 3.3.7、NumPy 1.26.4 和 PyYAML 6.0.3。建议先建立独立的 Python 环境。Windows PowerShell 将输出重定向到文件时，可先执行 `$env:PYTHONUTF8 = "1"`。

## 面板与事件判定

```bash
python -m pip install -r mobl_arms_index_choice_reaction/requirements.txt
python mobl_arms_index_choice_reaction/main.py --headless --max-steps 20
python mobl_arms_index_choice_reaction/test_button_press_state.py
```

面板入口从脚本同目录加载 XML 和 YAML。`main.py` 的步数按物理步累计，省略两个参数可打开交互窗口。面板接触演示、滑块事件状态机和学习环境各有独立入口。

## 单按钮 Gymnasium 环境

```bash
python -m pip install gymnasium==1.2.0
python mobl_arms_index_choice_reaction/verify_button_slider_env.py
```

检查包括观测是否有限、按下和释放能否按顺序各出现一次、超时是否截断回合。脚本使用手写的接近和回撤轨迹，结果写入 [Gymnasium 检查报告](../artifacts/trajectory_check_2026-10-27/button_slider_env_check.json)。环境每五个物理步接收一次二维动作。十五维观测包含关节状态、滑块状态、接触与事件阶段等信息。

## 策略训练与评估

```bash
python -m pip install -r mobl_arms_index_choice_reaction/requirements-rl.txt
python mobl_arms_index_choice_reaction/train_ppo_multiseed.py
```

脚本依次训练种子 2026、2027、2028。每个模型保存后重新加载，在测试种子 5200 至 5249 上评估。环境对初始接触进行重采样，达到最大尝试次数时单列无效初态。成功、峰值法向力与饱和计数按回合报告。复训会写入同名模型与报告，比较两次结果时请使用独立工作副本。[模型与评估结果](../artifacts/ppo_multiseed_2026-11-08/)

## 图 3 重建

```bash
python -m pip install "matplotlib>=3.6"
python scripts/plot_press_cycle.py
```

输入是 501 个等间隔物理状态采样。脚本读取在线状态机记录的事件时刻，把滑块位移换算为毫米，汇总指尖与按钮接触对的法向力，输出 [press-cycle.png](figures/press-cycle.png)。图 2 的两路相机回放来自独立的肩肘轨迹实验。[双轴回放元数据](../artifacts/trajectory_check_2026-10-21/dual_motion_render.json)

## 四按钮候选复核

| 阶段 | 代码 | 记录 |
| --- | --- | --- |
| 四自由度几何距离扫描 | [姿态采样](../mobl_arms_index_choice_reaction/diagnose_button_reachability_multidof.py) | [几何距离与候选关节角](../artifacts/trajectory_check_2026-11-08/button_reachability_multidof.json) |
| 近表面候选的动态跟踪 | [四关节 PD 与临时滑块](../mobl_arms_index_choice_reaction/verify_four_joint_slider_dynamics.py) | [慢速轨迹的位移与受力](../artifacts/trajectory_check_2026-11-08/four_joint_slider_dynamics_slow_pd.json) |
| 正面对齐与参考点审计 | [指尖网格核对](../mobl_arms_index_choice_reaction/audit_fingertip_mesh_reference.py) | [网格和标记点坐标](../artifacts/trajectory_check_2026-11-08/fingertip_mesh_reference.json) |

几何扫描直接设置关节位置，报告中的负距离表示穿透。动力学试验记录控制轨迹下的实际接触、行程和复位，使用这些量才能评价按压质量。

## 可选演示环境

摄像头指向演示使用 OpenCV 和带 `mediapipe.solutions.hands` 接口的 MediaPipe。安装视觉依赖后，在 [指向模块](../mobl_arms_index_pointing/) 内运行 `python evaluator.py`，该入口按当前目录读取配置和模型。两个 ROS 2 示例工作区的包名相同，应分别构建和加载。各自的入口见[关节与仿真数据采集](../ros2_ws/src/mujoco_ros_demo/README.md)和[指向感知示例](../mobl_arms_index_pointing/ros2_ws/src/mujoco_ros_demo/README.md)。
