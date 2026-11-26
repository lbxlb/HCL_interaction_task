# 面板模型关节结构核对

来源：当前面板 XML 与本地指向 XML。未修改正式模型；本地指向模型不能视为已验证的上游原版。

| 关节 | 所属 body ← 父 body | 有限位 | 演示增量驱动 | 与指向模型关节属性相同 |
|---|---|---|---|---|
| head_rot | head ← world | True | False | False |
| r_z | thorax ← world | True | False | True |
| sternoclavicular_r2 | clavicle ← thorax | False | False | True |
| sternoclavicular_r3 | clavicle ← thorax | False | False | True |
| unrotscap_r3 | clavphant ← clavicle | False | False | True |
| unrotscap_r2 | clavphant ← clavicle | False | False | True |
| acromioclavicular_r2 | scapula ← clavphant | False | False | True |
| acromioclavicular_r3 | scapula ← clavphant | False | False | True |
| acromioclavicular_r1 | scapula ← clavphant | False | False | True |
| unrothum_r1 | scapphant ← scapula | False | False | True |
| unrothum_r3 | scapphant ← scapula | False | False | True |
| unrothum_r2 | scapphant ← scapula | False | False | True |
| elv_angle | humphant ← scapphant | True | False | True |
| shoulder_elv | humphant1 ← humphant | True | False | True |
| shoulder1_r2 | humphant1 ← humphant | False | False | True |
| shoulder_rot | humerus ← humphant1 | True | True | True |
| elbow_flexion | ulna ← humerus | True | True | True |
| pro_sup | radius ← ulna | True | True | True |

## 本地指向模型中存在、面板模型中缺少的约束

下表是 XML 的 joint1、joint2 与 polycoef 原值；不能仅凭名称推断解剖学有效性。

| joint1 | joint2（无则约束单关节） | polycoef |
|---|---|---|
| sternoclavicular_r2 | shoulder_elv | 0 -0.242001 0 0 0 |
| sternoclavicular_r3 | shoulder_elv | 0 0.1025 0 0 0 |
| unrotscap_r2 | shoulder_elv | 0 0.242001 0 0 0 |
| unrotscap_r3 | shoulder_elv | 0 -0.1025 0 0 0 |
| acromioclavicular_r1 | shoulder_elv | 0 0.178 0 0 0 |
| acromioclavicular_r2 | shoulder_elv | 0 -0.0490002 0 0 0 |
| acromioclavicular_r3 | shoulder_elv | 0 0.396002 0 0 0 |
| unrothum_r1 | shoulder_elv | 0 -0.178 0 0 0 |
| unrothum_r2 | shoulder_elv | 0 0.0490002 0 0 0 |
| unrothum_r3 | shoulder_elv | 0 -0.396002 0 0 0 |
| shoulder1_r2 | elv_angle | 0 -1 0 0 0 |
| r_z | 无 | 0 0 0 0 0 |

若这些关系恢复且有效，按依赖关系计数剩余的独立候选关节：

head_rot, elv_angle, shoulder_elv, shoulder_rot, elbow_flexion, pro_sup

该计数不是约束雅可比秩验证，也不表示已经实现六自由度控制。
