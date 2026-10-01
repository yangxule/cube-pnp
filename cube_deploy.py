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
                        help=">0 时自动跑满该成功回合数后退出（成功率测试）；0=一直运行")
    parser.add_argument("--max_steps", type=int, default=400,
                        help="单回合同一尝试的最大控制步数（20Hz，400=20s）")
    args = parser.parse_args()

    seed = args.seed if args.seed >= 0 else None
    device = args.device
    print(f"device={device}, seed={seed}, ckpt={args.ckpt}")

    # ------------------------------------------------------------- policy
    dataset_metadata = LeRobotDatasetMetadata(args.repo_name, root=args.root)
    features = dataset_to_policy_features(dataset_metadata.features)
    output_features = {k: ft for k, ft in features.items()
                       if ft.type is FeatureType.ACTION}
    input_features = {k: ft for k, ft in features.items()
                      if k not in output_features}
    input_features.pop("observation.wrist_image", None)

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

    # ------------------------------------------------------------ rollout
    step = 0
    n_success = 0
    n_attempt = 0
    reset_episode()
    n_attempt += 1

    while env.env.is_viewer_alive():
        env.step_env()
        if not env.env.loop_every(HZ=20):
            continue

        # R 键：手动重开当前回合
        if env.env.is_key_pressed_once(key=glfw.KEY_R):
            print("Restart (R pressed)")
            reset_episode()
            n_attempt += 1
            step = 0
            env.render()
            continue

        # 构造观测（仅 agentview 主视角 + ee pose，与训练输入一致）
        state = env.get_ee_pose()
        image, _ = env.grab_image()
        image = Image.fromarray(image).resize((256, 256))
        image = img_transform(image)
        data = {
            'observation.state': torch.tensor([state]).to(device),
            'observation.image': image.unsqueeze(0).to(device),
            'task': [args.task_name],
            'timestamp': torch.tensor([step / 20.0]).to(device),
        }

        with torch.no_grad():
            action = policy.select_action(data)
        action = action[0].cpu().detach().numpy()
        env.step(action)
        env.render()
        step += 1

        # 成功判定
        if success_detector.check(env):
            n_success += 1
            cube = env.env.get_p_body(CUBE_BODY)
            print(f"[deploy] SUCCESS #{n_success} at step={step}, "
                  f"cube=({cube[0]:.3f},{cube[1]:.3f},{cube[2]:.3f})")
            if args.episodes > 0 and n_success >= args.episodes:
                break
            reset_episode()
            n_attempt += 1
            step = 0
            continue

        # 超时：本回合计为失败并重开
        if step >= args.max_steps:
            print(f"[deploy] attempt {n_attempt} TIMEOUT after {step} steps "
                  f"(success so far: {n_success})")
            reset_episode()
            n_attempt += 1
            step = 0

    env.env.close_viewer()
    if args.episodes > 0:
        print(f"Done. {n_success} success / {n_attempt} attempts "
              f"= {n_success / n_attempt:.1%} success rate")
    else:
        print(f"Done. {n_success} success over {n_attempt} attempts")


if __name__ == "__main__":
    main()
