# 实验数据索引

| 实验 | 数据入口 | 读取方式 |
| --- | --- | --- |
| 肩肘双轴运动 | [双视角 GIF](trajectory_check_2026-10-21/shoulder_elbow_dual_motion.gif) · [帧参数](trajectory_check_2026-10-21/dual_motion_render.json) | 两个视角对应同一时刻的关节姿态 |
| 单按钮按下与释放 | [逐步接触与状态机事件](trajectory_check_2026-10-22/button_press_state_online.json) | `samples` 为每 2 ms 的物理采样，`online_task.events` 记录带时间戳的事件 |
| Gymnasium 环境 | [手写动作与接口检查](trajectory_check_2026-10-27/button_slider_env_check.json) | `checks` 包含事件顺序、终止、超时和观测有效性 |
| 单按钮策略评估 | [多种子结果与模型](ppo_multiseed_2026-11-08/) | `ppo_multiseed_report.json` 的 `arms` 按训练种子分组，`evaluation` 按测试回合记录 |
| 四按钮目标扩展 | [几何与动力学报告](trajectory_check_2026-11-08/) | `button_reachability_multidof.json` 记录几何候选，`four_joint_slider_dynamics_slow_pd.json` 记录近表面候选的动态响应 |

同一日期的不同报告可能使用不同的滑块、控制器或起始姿态。比较结果时应同时核对实验条件、目标按钮和指标单位。部分环境构建器会读取已保存的扫描结果来恢复初始姿态，请保留相应目录结构。[复现指南](../docs/reproduction.md)
