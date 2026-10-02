#!/usr/bin/env python
"""部署训练好的 ACT 策略到 5cm 立方体 pick-and-place 环境。

迁移自根目录 4.deploy.py，改动：
  1. 场景为隔离的 cube_pnp/asset/example_scene_cube.xml，策略为 ckpt/act_cube。
  2. 成功判定使用 cube_collect_data 的接触式三条件（中心在盘内 + 接触盘面且
     静止 + 与夹爪分离），替代旧 env.check_success。
  3. reset 后等待立方体落稳再开始 rollout。
  4. 支持 R 键随时重开当前回合；--episodes N 可做自动成功率测试。

Usage:
    python cube_deploy.py                 # 固定 seed=0
    python cube_deploy.py --seed -1       # 随机物体位置
    python cube_deploy.py --episodes 10   # 自动跑 10 个成功回合后退出（成功率测试）
运行中按 R 键重开当前回合。
"""
import os
import sys
import argparse
from pathlib import Path

os.chdir(Path(__file__).resolve().parent)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import glfw
from PIL import Image
import torchvision
from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.common.datasets.utils import dataset_to_policy_features
from lerobot.common.policies.act.configuration_act import ACTConfig
from lerobot.common.policies.act.modeling_act import ACTPolicy
from lerobot.configs.types import FeatureType
from lerobot.common.datasets.factory import resolve_delta_timestamps

from mujoco_env.y_env import SimpleEnv
from cube_collect_data import (ContactGraspChecker, wait_until_settled,
                               CUBE_BODY, PLATE_BODY)


# ===========================================================================
# 成功判定（与采集端三条件一致）
# ===========================================================================
class PlaceSuccessDetector:
    """三条件连续 hold_ticks 个控制周期成立才判定成功，避免接触抖动误判。"""

    def __init__(self, checker, hold_ticks: int = 20, settle_dz: float = 0.002):
        self.checker = checker
        self.hold_ticks = hold_ticks
        self.settle_dz = settle_dz
        self.counter = 0
        self._z_hist = []

    def reset(self):
        self.counter = 0
        self._z_hist = []

    def check(self, env) -> bool:
        cube = env.env.get_p_body(CUBE_BODY)
        plate = env.env.get_p_body(PLATE_BODY)
        horiz = float(np.linalg.norm(cube[:2] - plate[:2]))
        on_plate = horiz < self.checker.plate_radius
        touching = len(self.checker.contacting_plate_geoms(env.env)) >= 1
        released = len(self.checker.contacting_gripper_geoms(env.env)) == 0

        self._z_hist.append(cube[2])
        if len(self._z_hist) > 10:
            self._z_hist.pop(0)
        settled = True
        if len(self._z_hist) == 10:
            settled = abs(self._z_hist[-1] - self._z_hist[0]) < self.settle_dz

        if on_plate and touching and released and settled:
            self.counter += 1
        else:
            self.counter = 0
        return self.counter >= self.hold_ticks


# ===========================================================================
# 7.2 失败模式分类
# ===========================================================================
LIFT_Z_MARGIN = 0.06   # cube 相对初始高度抬起超过 6cm 视为曾经抓起
TABLE_Z = 0.825        # 立方体在桌面静止时的中心高度（5cm 立方体半高 2.5cm）

def classify_failure(env, checker, cube_init_z: float, ever_lifted: bool) -> str:
    """回合超时未成功时，依据末态与过程标志分类失败原因。

    GRASP_FAIL     从未抓起（闭合后没带起来）
    DROPPED        曾抓起，末态掉回桌面且离盘远（搬运途中脱落）
    PLACE_OFFSET   立方体已在盘面高度但水平偏出盘子半径
    NOT_RELEASED   到达盘位但夹爪仍与立方体接触（没松开/松开失败）
    TIMEOUT_OTHER  其他（卡住、姿态异常等）
    """
    cube = env.env.get_p_body(CUBE_BODY)
    plate = env.env.get_p_body(PLATE_BODY)
    horiz = float(np.linalg.norm(cube[:2] - plate[:2]))
    npc = len(checker.contacting_plate_geoms(env.env))
    ngc = len(checker.contacting_gripper_geoms(env.env))

    if not ever_lifted:
        return "GRASP_FAIL"
    if npc >= 1 and ngc > 0:
        return "NOT_RELEASED"
    if (npc >= 1 or abs(cube[2] - TABLE_Z) < 0.04) and horiz >= checker.plate_radius:
        return "PLACE_OFFSET"
    if cube[2] < TABLE_Z + 0.04 and horiz >= checker.plate_radius:
        return "DROPPED"
    return "TIMEOUT_OTHER"


# ===========================================================================
# Main
# ===========================================================================
def main():
    parser = argparse.ArgumentParser(description="Deploy ACT policy on cube env")
    parser.add_argument("--seed", type=int, default=0,
                        help="环境随机种子，传负数（如 -1）则随机位置")
    parser.add_argument("--root", type=str, default="./demo_data")
    parser.add_argument("--repo_name", type=str, default="omy_cube_pnp")
    parser.add_argument("--task_name", type=str, default="Put cube on the plate")
    parser.add_argument("--ckpt", type=str, default="./ckpt/act_cube")
    parser.add_argument("--xml_path", type=str, default="./asset/example_scene_cube.xml")
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--episodes", type=int, default=0,
                        help="演示模式：跑满该成功回合数后退出；0=一直运行")
    parser.add_argument("--eval", type=int, default=0, metavar="N",
                        help="评测模式：固定跑 N 个回合（成败均计数），输出成功率与失败分类")
    parser.add_argument("--max_steps", type=int, default=400,
                        help="单回合最大控制步数（20Hz，400=20s）")
    args = parser.parse_args()

    seed = args.seed if args.seed >= 0 else None
    device = args.device
    eval_mode = args.eval > 0
    print(f"device={device}, seed={seed}, ckpt={args.ckpt}, "
          f"mode={'EVAL ' + str(args.eval) + ' eps' if eval_mode else 'demo'}")

    # ------------------------------------------------------------- policy
    dataset_metadata = LeRobotDatasetMetadata(args.repo_name, root=args.root)
    features = dataset_to_policy_features(dataset_metadata.features)
    output_features = {k: ft for k, ft in features.items()
                       if ft.type is FeatureType.ACTION}
    input_features = {k: ft for k, ft in features.items()
                      if k not in output_features}
    # 以 checkpoint 自身 config 为准：双视角模型用 wrist，旧单视角模型 pop 掉
    import json
    with open(Path(args.ckpt) / "config.json") as f:
        ckpt_cfg = json.load(f)
    use_wrist = "observation.wrist_image" in ckpt_cfg["input_features"]
    if not use_wrist:
        input_features.pop("observation.wrist_image", None)
    print(f"camera inputs: {'agentview + wrist' if use_wrist else 'agentview only'}")

    # 部署用 temporal ensemble（n_action_steps=1）平滑轨迹
    cfg = ACTConfig(input_features=input_features,
                    output_features=output_features,
                    chunk_size=10, n_action_steps=1,
                    temporal_ensemble_coeff=0.9)
    resolve_delta_timestamps(cfg, dataset_metadata)
    policy = ACTPolicy.from_pretrained(args.ckpt, config=cfg,
                                       dataset_stats=dataset_metadata.stats)
    policy.to(device)
    policy.eval()

    # --------------------------------------------------------------- env
    env = SimpleEnv(args.xml_path, action_type='joint_angle', seed=seed)
    checker = ContactGraspChecker(env.env, min_geoms=2)
    success_detector = PlaceSuccessDetector(checker)
    img_transform = torchvision.transforms.ToTensor()

    def reset_episode():
        env.reset(seed=seed)
        wait_until_settled(env)
        policy.reset()
        success_detector.reset()
        return float(env.env.get_p_body(CUBE_BODY)[2])

    # ------------------------------------------------------------ rollout
    step = 0
    n_success = 0
    n_attempt = 0
    fail_categories = {}          # 失败模式 -> 次数
    ever_lifted = False
    cube_init_z = reset_episode()
    n_attempt += 1
    n_target = args.eval if eval_mode else args.episodes

    while env.env.is_viewer_alive():
        env.step_env()
        if not env.env.loop_every(HZ=20):
            continue

        # R 键：手动重开当前回合（eval 模式下禁用，保证分母准确）
        if not eval_mode and env.env.is_key_pressed_once(key=glfw.KEY_R):
            print("Restart (R pressed)")
            cube_init_z = reset_episode()
            n_attempt += 1
            step = 0
            ever_lifted = False
            env.render()
            continue

        # 构造观测（agentview，双视角模型时另加 wrist；+ ee pose，与训练输入一致）
        state = env.get_ee_pose()
        image, wrist_image = env.grab_image()
        image = img_transform(Image.fromarray(image).resize((256, 256)))
        data = {
            'observation.state': torch.tensor([state]).to(device),
            'observation.image': image.unsqueeze(0).to(device),
            'task': [args.task_name],
            'timestamp': torch.tensor([step / 20.0]).to(device),
        }
        if use_wrist:
            wrist_image = img_transform(
                Image.fromarray(wrist_image).resize((256, 256)))
            data['observation.wrist_image'] = wrist_image.unsqueeze(0).to(device)

        with torch.no_grad():
            action = policy.select_action(data)
        action = action[0].cpu().detach().numpy()
        env.step(action)
        env.render()
        step += 1

        # 过程标志：cube 是否曾被带离初始高度
        if not ever_lifted and \
                float(env.env.get_p_body(CUBE_BODY)[2]) > cube_init_z + LIFT_Z_MARGIN:
            ever_lifted = True

        episode_done = False
        episode_ok = False

        # 成功判定
        if success_detector.check(env):
            episode_done = True
            episode_ok = True
            n_success += 1
            cube = env.env.get_p_body(CUBE_BODY)
            print(f"[eval {n_attempt}/{n_target}] SUCCESS at step={step}, "
                  f"cube=({cube[0]:.3f},{cube[1]:.3f},{cube[2]:.3f})")

        # 超时：本回合计为失败
        elif step >= args.max_steps:
            episode_done = True
            reason = classify_failure(env, checker, cube_init_z, ever_lifted)
            fail_categories[reason] = fail_categories.get(reason, 0) + 1
            cube = env.env.get_p_body(CUBE_BODY)
            print(f"[eval {n_attempt}/{n_target}] FAIL [{reason}] step={step}, "
                  f"ever_lifted={ever_lifted}, cube=({cube[0]:.3f},{cube[1]:.3f},"
                  f"{cube[2]:.3f})")

        if not episode_done:
            continue

        # eval 模式：跑满固定回合数退出；demo 模式：跑满成功数退出
        if (eval_mode and n_attempt >= n_target) or \
           (not eval_mode and n_target > 0 and n_success >= n_target):
            break

        cube_init_z = reset_episode()
        n_attempt += 1
        step = 0
        ever_lifted = False

    env.env.close_viewer()

    if eval_mode:
        n_fail = n_attempt - n_success
        print("\n" + "=" * 60)
        print(f"EVAL RESULT: {n_success}/{n_attempt} success "
              f"= {n_success / n_attempt:.1%}")
        if n_fail:
            print("failure breakdown:")
            for k in ["GRASP_FAIL", "DROPPED", "PLACE_OFFSET",
                      "NOT_RELEASED", "TIMEOUT_OTHER"]:
                if fail_categories.get(k):
                    print(f"  {k:14s} {fail_categories[k]:3d}  "
                          f"({fail_categories[k] / n_fail:.0%} of failures, "
                          f"{fail_categories[k] / n_attempt:.0%} of all)")
        print("=" * 60)
    elif n_target > 0:
        print(f"Done. {n_success} success / {n_attempt} attempts")
    else:
        print(f"Done. {n_success} success over {n_attempt} attempts")


if __name__ == "__main__":
    main()
