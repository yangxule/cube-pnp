# Cube Pick-and-Place with Contact Detection

> 用 5cm 立方体替代原杯子，引入**接触检测判定抓取**，配合自适应闭合与软垫夹爪，
> 实现「自动数据采集 → ACT 策略训练 → 部署」的完整闭环，并通过 DAgger 恢复数据将随机布局成功率从 2% 提升至 32%。

---

## 一、项目任务

### 目标

- 抓取对象：5cm 立方体（原杯子任务的物体替换）
- 抓取判定：接触检测（≥2 个夹爪/软垫 geom 与立方体接触）替代旧的"抬起看高度"反推
- 闭环链路：FSM 自动采集演示 → ACT 模仿学习 → 策略部署
- 放置判定：三条件同时满足（中心在盘半径内 + 接触盘面且静止 + 与夹爪零接触）

### 关键参数（已验证）

| 参数 | 值 |
|---|---|
| 立方体尺寸 | 5cm（half-size 0.025） |
| 立方体 friction / density | 1.5 / 1000 |
| 软垫 friction | 2.0 |
| 接触检测阈值 min_geoms | 2 |
| 自适应闭合触发 | 从 0.5 逐步 +0.05，接触后 +0.15 夹紧 |
| 采集频率 | 20 Hz |
| 动作空间 | 绝对关节角（6 关节 + 夹爪命令） |

### 目录结构

```
cube_pnp/
├── asset/                          # 隔离的模型资产（独立于根目录杯子任务）
│   ├── example_scene_cube.xml      # 场景
│   ├── robotis_omy/omy_cube.xml    # 含软垫的夹爪模型
│   └── objaverse/cube_5cm/         # 5cm 立方体模型
├── cube_collect_data.py            # FSM 自动采集
├── cube_visualize.py               # 数据可视化
├── cube_train.py                   # ACT 训练
├── cube_deploy.py                  # 部署评测
├── cube_dagger_collect.py          # DAgger 恢复数据采集
├── demo_data/                      # 采集的演示数据（lerobot 格式）
├── ckpt/                           # 训练好的模型
│   ├── act_cube/                   # 基线模型（100 集随机数据）
│   └── act_cube_dagger/            # DAgger 模型
└── todolist.md                     # 项目规划与进度
```

---

## 二、环境部署

```bash
conda activate lerobot-mujoco-tutorial
cd /home/xl/lerobot-mujoco-tutorial/cube_pnp
```

依赖：`mujoco`、`torch`、`lerobot`、`torchvision`、`numpy`、`Pillow`。

> 注意：模型资产（asset/）、数据集（demo_data/）、checkpoint（ckpt/）均已 gitignore，需本地生成。

---

## 三、操作步骤

### 步骤 1：环境资产迁移与隔离

已完成。`cube_pnp/asset/` 下包含独立的 `omy_cube.xml`（含软垫）、`example_scene_cube.xml`、`cube_5cm` 模型。根目录杯子任务不受影响。

### 步骤 2：数据采集

```bash
python cube_collect_data.py --num_demo 100 --seed -1 --root ./demo_data
```

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--num_demo` | 50 | 采集 episode 数 |
| `--seed` | -1 | 随机种子。-1=随机布局；固定值=可复现的固定位置 |
| `--root` | `./demo_data` | 数据集输出目录 |
| `--xml_path` | `./asset/example_scene_cube.xml` | 场景文件 |
| `--min_contact_geoms` | 2 | 判定抓取成功的最少接触 geom 数 |
| `--max_retries` | 30 | 单 episode 最大失败重试次数 |
| `--delete-old` | False | root 已存在时直接删除重建 |

采集逻辑：FSM 状态机驱动，接触检测判定抓取成功，自适应闭合避免挤飞/滑脱。记录字段：`observation.image`、`observation.wrist_image`、`observation.state`(末端位姿)、`action`(绝对关节角+夹爪)、`obj_init`。

### 步骤 3：数据可视化

```bash
python cube_visualize.py --root ./demo_data --mode plot      # 导出帧+关节曲线
python cube_visualize.py --root ./demo_data --mode replay    # GT 开环回放
python cube_visualize.py --root ./demo_data --mode viewer    # 双相机 viewer 回放
python cube_visualize.py --root ./demo_data --mode stats     # 统计验证
```

### 步骤 4：训练 ACT 策略

```bash
python cube_train.py --steps 4000 --root ./demo_data --ckpt ./ckpt/act_cube --val_ratio 0.1
```

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--steps` | 1500 | 训练总更新步数（1 step = 1 个 batch 的前向+反向+参数更新） |
| `--root` | `./demo_data` | 数据集目录 |
| `--ckpt` | `./ckpt/act_cube` | 模型保存目录 |
| `--val_ratio` | 0.1 | 验证集占总 episode 的比例（按 episode 划分） |
| `--batch_size` | 64 | 每次更新喂入的帧数 |
| `--lr` | 1e-4 | Adam 学习率 |
| `--log_freq` | 100 | 每 N 步打印 train loss |
| `--val_freq` | 500 | 每 N 步打印 val loss |
| `--num_workers` | 4 | DataLoader 并行加载进程数 |
| `--no_wrist` | False | 加此参数则退回单视角（仅 agentview）；默认双相机 |
| `--no-cuda` | False | 加此参数强制 CPU；否则自动用 cuda |

ACT 配置：`chunk_size=10, n_action_steps=10`，默认双视角（agentview + wrist）。

### 步骤 5：部署评测

```bash
python cube_deploy.py --eval 100 --seed -1 --ckpt ./ckpt/act_cube
```

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--ckpt` | `./ckpt/act_cube` | 模型路径 |
| `--seed` | -1 | -1=随机布局；固定值=固定位置 |
| `--eval` | 0 | >0 时自动跑满该回合数后退出（成功率测试）；0=一直运行 |
| `--max_steps` | 400 | 单回合最大步数（20Hz） |
| `--task_name` | `Put cube on the plate` | 语言指令 |

部署用 temporal ensemble（`n_action_steps=1, coeff=0.9`）平滑轨迹。成功判定：三条件连续 20 个控制周期成立。

### 步骤 6（进阶）：DAgger 恢复数据采集

```bash
python cube_dagger_collect.py --num_demo 50 --ckpt ./ckpt/act_cube --root ./demo_data
```

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--num_demo` | 20 | 成功采集的 episode 数 |
| `--ckpt` | `./ckpt/act_cube` | 直播策略的 checkpoint |
| `--root` | `./demo_data` | 数据集 root（已存在则追加） |
| `--stats_root` | `./demo_data` | 策略归一化统计来源 |
| `--policy_max_steps` | 400 | 策略阶段单回合最大步数 |
| `--grasp_deadline` | 120 | 超过该步数仍未抬起立方体则接管 |
| `--hold_deadline` | 250 | 夹持状态下超过该步数未完成放置则接管 |
| `--fsm_max_steps` | 1200 | FSM 恢复阶段最大步数 |

DAgger 原理：策略先跑，检测偏离（夹空/掉落/超时）后 FSM 从当前状态接管完成，把整段「策略+恢复」轨迹录成新 demo，教策略从自身误差状态恢复。数据集格式与普通采集完全一致，可追加到同一 root 后用 `cube_train.py` 重训。

---

## 四、评测结果

### 基线模型 act_cube

> 100 集随机位置数据 + 4000 步训练，单视角。

```
命令: python cube_deploy.py --eval 100 --seed -1 --ckpt ./ckpt/act_cube
成功率: 2/100 = 2.0%
```

| 失败类型 | 次数 | 占比 | 含义 |
|---|---|---|---|
| GRASP_FAIL | 49 | 49% | 从未抓起（夹空/推跑） |
| PLACE_OFFSET | 44 | 44% | 抓起但放置偏离盘子 |
| DROPPED | 4 | 4% | 搬运中掉落飞出桌面 |
| TIMEOUT_OTHER | 1 | 1% | 抬起但超时未完成 |

### DAgger 模型 act_cube_dagger

> 基线模型 + DAgger 接管数据继续训练。

```
命令: python cube_deploy.py --eval 100 --seed -1 --ckpt ./ckpt/act_cube_dagger
成功率: 32/100 = 32.0%
```

| 失败类型 | 次数 | 占比 | 含义 |
|---|---|---|---|
| GRASP_FAIL | 45 | 45% | 从未抓起（夹空/推跑） |
| PLACE_OFFSET | 15 | 15% | 抓起但放置偏离盘子 |
| TIMEOUT_OTHER | 8 | 8% | 抬起但超时未完成 |
| DROPPED | 0 | 0% | 无掉落 |

---

## 五、两者对比

| 指标 | 基线 act_cube | DAgger act_cube_dagger | 变化 |
|---|---|---|---|
| **成功率** | 2.0% | **32.0%** | **↑16 倍** |
| GRASP_FAIL | 49 | 45 | -4（持平，仍为主要瓶颈） |
| PLACE_OFFSET | 44 | 15 | **-29（DAgger 教会放置修正）** |
| DROPPED | 4 | 0 | **-4（搬运稳定性提升）** |
| TIMEOUT_OTHER | 1 | 8 | +7（抬起后悬停未完成） |

### 结论

- **DAgger 有效**：成功率 2% → 32%，对放置环节（PLACE_OFFSET -29）和搬运稳定性（DROPPED 归零）提升显著。
- **剩余瓶颈是抓取阶段**（GRASP_FAIL 45%）：DAgger 对这部分改善有限，因为夹空后立方体没动，恢复段仍从桌面重新抓取，与原始数据重合度高，未带来太多新信息。
- **后续优化方向**：
  1. 改用末端增量（delta-eef）动作空间，直击闭合精度根因
  2. 第二轮 DAgger，降低 `grasp_deadline` 提前触发接管，采更多"夹空瞬间→恢复"轨迹
  3. 双视角（wrist 相机）模型为闭合时机提供毫米级视觉信息

---

## 六、参考

- 项目规划与进度：[todolist.md](./todolist.md)
- 采集 IK 原理：阻尼最小二乘（DLS），λ=0.01，从当前关节角热启动迭代
- 接触检测：未配置接触力传感器，通过 mujoco geom 接触数判定（软垫 contype/conaffinity=1）
