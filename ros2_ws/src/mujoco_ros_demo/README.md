# MuJoCo 关节状态与仿真数据采集

本包在 ROS 2 中运行独立的 MuJoCo 示例。发布节点读取 `config/humanoid.xml` 并将前六个广义位置发送至 `/mujoco/joint_angles`；订阅节点读取该话题。采集节点另行推进自己的模型，发布物体位置、关节位置和姿态。

在装有 ROS 2 Humble、colcon 与 MuJoCo 的 Ubuntu 22.04 环境中，从仓库根目录进入此工作区。现有 launch 文件为发布与采集节点激活 `~/miniconda3` 下的 `mjoco_py310` 环境。

```bash
source /opt/ros/humble/setup.bash
cd ros2_ws
colcon build --symlink-install
source install/setup.bash
ros2 launch mujoco_ros_demo main.launch.py
```

面板策略评估由独立的 Gymnasium 环境运行。[研究首页](../../../README.md)
