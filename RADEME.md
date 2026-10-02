# Cube Pick-and-Place with Contact Detection

基于 ACT 模仿学习与 MuJoCo 仿真的立方体抓取-放置任务。采用接触检测判定抓取状态、
自适应夹爪闭合与软垫夹爪设计，实现「自动数据采集 → 策略训练 → 闭环部署」全流程，
并通过 DAgger 恢复数据机制将随机布局下的成功率从 2% 提升至 32%。

---

## 一、项目背景

本项目基于 [lerobot-mujoco-tutorial](https://github.com/jeongeun980906/lerobot-mujoco-tutorial) 二次开发。
原教程在 MuJoCo 仿真中实现了 ROBOTIS OpenMANIPULATOR-Y 机械臂的**马克杯抓取-放置**任务，
技术路线为：FSM/遥操作采集演示数据（LeRobot 数据集格式）→ ACT 模仿学习训练 → 闭环部署。

本项目将抓取对象由马克杯替换为 **5cm 光滑立方体**，并在抓取策略、感知方式与模型管理方面做了以下增强：

| 增强项 | 实现 |
|---|---|
| 抓取对象 | 5cm 光滑立方体（针对点接触易滑脱问题） |
| 夹爪 | 指尖附加软垫（friction=2.0），增大接触面积 |
| 抓取判定 | **接触检测**：≥2 个夹爪/软垫 geom 与立方体接触 |
| 闭合策略 | **自适应闭合**：从 0.5 逐步逼近，接触触发后 +0.15 夹紧 |
| 放置判定 | 三条件联合判定：中心在盘内 + 接触盘面且静止 + 与夹爪零接触 |
| 鲁棒性增强 | **DAgger**：策略偏离后由 FSM 接管，记录"偏离→恢复"轨迹 |
| 模型资产 | 完全隔离于 `cube_pnp/asset/`，与原任务互不影响 |

原教程脚本（根目录 `1~8.py` 及 `.ipynb`）保持不变，本项目全部代码与资产独立存放于 `cube_pnp/` 目录。

---

## 二、项目任务

### 目标

- 抓取对象：5cm 立方体
- 抓取判定：基于 geom 接触数（≥2）的实时接触检测
- 技术闭环：FSM 自动采集演示 → ACT 模仿学习 → 策略闭环部署
- 放置判定：三条件同时满足（立方体中心在盘半径内 + 接触盘面且静止 + 与夹爪零接触）

### 关键参数（已验证）

| 参数 | 值 |
|---|---|
| 立方体尺寸 | 5cm（half-size 0.025） |
| 立方体 friction / density | 1.5 / 1000 |
| 软垫 friction | 2.0 |
| 接触检测阈值 min_geoms | 2 |
| 自适应闭合 | 起始 0.5，每 8 tick +0.05，接触后 +0.15 夹紧 |
| 控制频率 | 20 Hz |
| 动作空间 | 绝对关节角（6 关节 + 夹爪命令，共 7 维） |

### 目录结构

```
cube_pnp/
├── asset/                          # 隔离的模型资产
│   ├── example_scene_cube.xml      # 场景定义
│   ├── robotis_omy/omy_cube.xml    # 含软垫的机械臂模型
│   └── objaverse/cube_5cm/         # 5cm 立方体模型
├── cube_collect_data.py            # FSM 自动数据采集
├── cube_visualize.py               # 数据可视化与回放
├── cube_train.py                   # ACT 策略训练
├── cube_deploy.py                  # 闭环部署与评测
├── cube_dagger_collect.py          # DAgger 恢复数据采集
├── demo_data/                      # 演示数据集（LeRobot 格式，gitignore）
├── ckpt/                           # 模型 checkpoint（gitignore）
│   ├── act_cube/                   # 基线模型
│   └── act_cube_dagger/            # DAgger 增强模型
└── todolist.md                     # 项目规划与进度追踪
```

---

## 三、环境部署

Python 依赖与原项目完全一致，无需安装额外依赖包。

- **Python 版本：3.10**
- 环境管理：本项目使用 conda 环境

```bash
conda activate your_env
cd ~/lerobot-mujoco-tutorial/cube_pnp
```

依赖安装请参考原仓库 [lerobot-mujoco-tutorial](https://github.com/jeongeun980906/lerobot-mujoco-tutorial) 的说明
（核心依赖：`mujoco`、`torch`、`lerobot`、`torchvision`、`numpy`、`Pillow`）。

> 说明：`asset/`、`demo_data/`、`ckpt/` 已加入 gitignore，需通过下方流程在本地生成。

---

## 四、操作流程

### 步骤 1：数据采集

```bash
python cube_collect_data.py --num_demo 100 --seed -1 --root ./demo_data
```

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--num_demo` | 50 | 采集 episode 数量 |
| `--seed` | -1 | 随机种子；-1 为随机布局，固定值为可复现的固定位置 |
| `--root` | `./demo_data` | 数据集输出目录 |
| `--xml_path` | `./asset/example_scene_cube.xml` | 场景文件路径 |
| `--min_contact_geoms` | 2 | 判定抓取成功的最少接触 geom 数 |
| `--max_retries` | 30 | 单 episode 最大重试次数 |
| `--delete-old` | False | 数据集目录已存在时直接删除重建 |

采集流程由 FSM 状态机驱动，通过接触检测实时判定抓取成功，自适应闭合避免过压挤飞或欠压滑脱。
记录字段：`observation.image`、`observation.wrist_image`、`observation.state`（末端位姿）、`action`（绝对关节角 + 夹爪命令）、`obj_init`。

### 步骤 2：数据可视化

```bash
python cube_visualize.py --root ./demo_data --mode plot      # 导出帧图像与关节曲线
python cube_visualize.py --root ./demo_data --mode replay    # GT 开环回放验证
python cube_visualize.py --root ./demo_data --mode viewer    # 双相机 viewer 回放
python cube_visualize.py --root ./demo_data --mode stats     # 数据集统计
```

### 步骤 3：训练 ACT 策略

```bash
python cube_train.py --steps 4000 --root ./demo_data --ckpt ./ckpt/act_cube --val_ratio 0.1
```

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--steps` | 1500 | 训练总更新步数（1 step = 1 个 batch 的前向 + 反向 + 参数更新） |
| `--root` | `./demo_data` | 数据集目录 |
| `--ckpt` | `./ckpt/act_cube` | 模型保存目录 |
| `--val_ratio` | 0.1 | 验证集 episode 占比（按 episode 划分） |
| `--batch_size` | 64 | 单次更新的样本帧数 |
| `--lr` | 1e-4 | Adam 学习率 |
| `--log_freq` | 100 | train loss 打印间隔（步） |
| `--val_freq` | 500 | val loss 评估间隔（步） |
| `--num_workers` | 4 | DataLoader 并行加载进程数 |
| `--no_wrist` | False | 指定后退回单视角（仅 agentview）；默认双相机输入 |
| `--no-cuda` | False | 指定后强制使用 CPU；默认自动检测 GPU |

ACT 配置：`chunk_size=10, n_action_steps=10`；默认输入为双视角（agentview + wrist）。

### 步骤 4：闭环部署与评测

```bash
python cube_deploy.py --eval 100 --seed -1 --ckpt ./ckpt/act_cube
```

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--ckpt` | `./ckpt/act_cube` | 模型 checkpoint 路径 |
| `--seed` | -1 | -1 为随机布局；固定值为固定位置 |
| `--eval` | 0 | >0 时自动运行指定回合数并统计成功率；0 为持续运行 |
| `--max_steps` | 400 | 单回合最大控制步数（20Hz） |
| `--task_name` | `Put cube on the plate` | 语言指令 |

部署阶段使用 temporal ensemble（`n_action_steps=1, coeff=0.9`）平滑动作序列；
成功判定要求三条件在连续 20 个控制周期内同时成立。

### 步骤 5（进阶）：DAgger 恢复数据采集

```bash
python cube_dagger_collect.py --num_demo 50 --ckpt ./ckpt/act_cube --root ./demo_data
```

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--num_demo` | 20 | 成功采集的 episode 数 |
| `--ckpt` | `./ckpt/act_cube` | 直播策略的 checkpoint |
| `--root` | `./demo_data` | 数据集目录（已存在则追加） |
| `--stats_root` | `./demo_data` | 策略归一化统计来源 |
| `--policy_max_steps` | 400 | 策略阶段单回合最大步数 |
| `--grasp_deadline` | 120 | 超过该步数未抬起立方体则触发接管 |
| `--hold_deadline` | 250 | 夹持状态下超过该步数未完成放置则触发接管 |
| `--fsm_max_steps` | 1200 | FSM 恢复阶段最大步数 |

DAgger 机制：策略闭环执行，检测到偏离（夹空 / 掉落 / 超时）后由 FSM 从当前状态接管并完成剩余任务，
将完整的「策略 + 恢复」轨迹记录为新 demo，使策略学习从自身误差状态中恢复的能力。
数据格式与普通采集完全一致，追加至同一数据集后可直接使用 `cube_train.py` 重新训练。

---

## 五、评测结果

### 基线模型 act_cube

> 训练配置：100 集随机位置数据，4000 步，单视角输入。

```
python cube_deploy.py --eval 100 --seed -1 --ckpt ./ckpt/act_cube
成功率：2/100 = 2.0%
```

| 失败类型 | 次数 | 占比 | 说明 |
|---|---|---|---|
| GRASP_FAIL | 49 | 49% | 未能抓起立方体（夹空 / 推跑） |
| PLACE_OFFSET | 44 | 44% | 抓取成功但放置偏离盘面 |
| DROPPED | 4 | 4% | 搬运过程中立方体掉落 |
| TIMEOUT_OTHER | 1 | 1% | 抬起后超时未完成放置 |

### DAgger 模型 act_cube_dagger

> 训练配置：基线数据集 + DAgger 接管数据，联合训练。

```
python cube_deploy.py --eval 100 --seed -1 --ckpt ./ckpt/act_cube_dagger
成功率：32/100 = 32.0%
```

| 失败类型 | 次数 | 占比 | 说明 |
|---|---|---|---|
| GRASP_FAIL | 45 | 45% | 未能抓起立方体 |
| PLACE_OFFSET | 15 | 15% | 抓取成功但放置偏离盘面 |
| TIMEOUT_OTHER | 8 | 8% | 抬起后超时未完成放置 |
| DROPPED | 0 | 0% | 无掉落 |

---

## 六、结果对比

| 指标 | 基线 act_cube | DAgger act_cube_dagger | 变化 |
|---|---|---|---|
| **成功率** | 2.0% | **32.0%** | **提升 16 倍** |
| GRASP_FAIL | 49 | 45 | -4（抓取阶段仍为瓶颈） |
| PLACE_OFFSET | 44 | 15 | **-29（放置修正能力显著增强）** |
| DROPPED | 4 | 0 | **-4（搬运稳定性提升）** |
| TIMEOUT_OTHER | 1 | 8 | +7 |

### 结论

- **DAgger 机制有效**：成功率由 2% 提升至 32%。放置阶段（PLACE_OFFSET 44→15）与搬运稳定性（DROPPED 4→0）改善显著，
  表明恢复数据有效扩展了策略的吸引域。
- **剩余瓶颈集中于抓取阶段**（GRASP_FAIL 45%）：夹空后立方体位姿基本不变，FSM 接管后的恢复轨迹与原始演示重合度较高，
  未为抓取阶段引入足够的新状态分布。
- **后续优化方向**：
  1. 采用末端增量（delta-eef）动作空间，从动作表征层面提升闭合精度
  2. 第二轮 DAgger：降低 `grasp_deadline` 提前触发接管，扩充抓取阶段的恢复样本
  3. 引入 wrist 相机双视角输入，为闭合时机判断提供近距离视觉信息
  4. 部署端动作平滑（低通滤波 / 指数平滑），抑制闭合瞬间的高频抖动

---

## 七、技术参考

- 项目规划与进度：[todolist.md](./todolist.md)
- IK 求解：阻尼最小二乘（DLS），λ=0.01，基于当前关节角热启动迭代
- 接触检测：基于 MuJoCo geom 接触对计数实现（未配置接触力传感器；软垫 contype/conaffinity=1 参与碰撞）
