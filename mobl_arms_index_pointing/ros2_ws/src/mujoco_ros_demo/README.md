# 上肢指向感知的 ROS 2 示例

该工作区包含关节发布与订阅节点，以及指向任务的采集和感知节点。采集节点提供手指关节、指尖和目标球的位置，感知节点订阅这些话题。默认 launch 仅启动关节发布与订阅。

使用 ROS 2 Humble 在本工作区单独构建。发布节点的启动脚本使用 `~/miniconda3` 下的 `mjoco_py310` 环境。

```bash
source /opt/ros/humble/setup.bash
cd mobl_arms_index_pointing/ros2_ws
colcon build --symlink-install
source install/setup.bash
ros2 launch mujoco_ros_demo main.launch.py
```

采集节点另需摄像头与指向仿真环境。复制安装时可将 `HCL_POINTING_DIR` 设为含 `simulator.py`、模型和配置的指向模块绝对路径。根目录另有同名 ROS 2 包，包含另一组采集节点。[研究首页](../../../../README.md)
