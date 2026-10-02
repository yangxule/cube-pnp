#!/usr/bin/env python
"""DAgger 式恢复数据采集：策略先跑，跑偏后 FSM 从当前状态接管完成，
把整段「策略 + 恢复」轨迹录成新 demo，直接教策略从自身误差状态恢复。

原理（DAgger）：
  1. 策略阶段：ACT 策略按部署配置（joint_angle + temporal ensemble 0.9）直播，
     逐帧记录真实观测与下发动作（与 cube_collect_data 完全相同的数据结构）。
  2. 接管：检测到偏离后把 env.action_type 切到 eef_pose，同步 p0/R0 到当前
     tcp，FSM 从当前状态重新规划并完成放置：
       - 立方体落回桌面（曾抬起/夹持后掉落） -> FSM 重新抓取
       - 超过 grasp-deadline 仍从未抬起      -> FSM 重新抓取
       - 夹持状态下超过 hold-deadline        -> FSM 直接从 TRANSPORT 继续
       - 策略阶段达到 policy-max-steps       -> 按当前状态接管或判负
  3. 恢复段的「偏离状态 + 专家动作」是扩大策略吸引域的关键数据；策略独立
     完成的回合直接保存（等价普通 demo）。失败回合清空缓存重试。

数据集格式与 cube_collect_data 完全一致，追加进同一 root 后可直接用
cube_train.py 继续训练（无需任何改动）。

Usage:
    python cube_dagger_collect.py --num_demo 20        # 追加到 ./demo_data（默认）
    python cube_dagger_collect.py --num_demo 50 --root ./demo_data_dagger --delete-old
"""
import os
import sys
import argparse
import json
from pathlib import Path

os.chdir(Path(__file__).resolve().parent)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
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
                               CubePickPlaceFSM, State, build_dataset,
                               CUBE_BODY, PLATE_BODY)

LIFT_Z_MARGIN = 0.06    # 与部署端一致的抬起判定（cube 相对初始高度抬起 6cm）
HOLD_MIN_GEOMS = 2      # 判定「夹持中」所需的最少夹爪接触 geom 数（与采集端一致）


class PlaceSuccessDetector:
    """三条件成功判定（与 cube_deploy.PlaceSuccessDetector 一致）：
    中心在盘内 + 接触盘面且静止 + 与夹爪分离，连续 hold_ticks 个周期成立。"""

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


def is_holding(env, checker, cube_init_z: float) -> bool:
    """立方体当前是否被夹爪夹住（接触数足够且离桌）。"""
    cube_z = float(env.env.get_p_body(CUBE_BODY)[2])
    n_grip = len(checker.contacting_gripper_geoms(env.env))
    return n_grip >= HOLD_MIN_GEOMS and cube_z > cube_init_z + 0.01


def cube_dropped(env, checker, cube_init_z: float, ever_lifted: bool) -> bool:
    """曾抬起，但现在立方体落回桌面且已与夹爪分离。"""
    if not ever_lifted:
        return False
    cube_z = float(env.env.get_p_body(CUBE_BODY)[2])
    return (cube_z < cube_init_z + 0.01
            and len(checker.contacting_gripper_geoms(env.env)) == 0)


def main():
    parser = argparse.ArgumentParser(description="DAgger recovery demo collection")
    parser.add_argument("--num_demo", type=int, default=20,
                        help="成功采集的 episode 数")
    parser.add_argument("--seed", type=int, default=-1,
                        help="随机种子，负数（如 -1）则随机布局")
    parser.add_argument("--root", type=str, default="./demo_data",
                        help="数据集 root；已存在则追加，--delete-old 可清空重建")
    parser.add_argument("--repo_name", type=str, default="omy_cube_pnp")
    parser.add_argument("--task_name", type=str, default="Put cube on the plate")
    parser.add_argument("--ckpt", type=str, default="./ckpt/act_cube")
    parser.add_argument("--stats_root", type=str, default="./demo_data",
                        help="策略归一化统计来源（= 训练该 ckpt 的数据集 root）")
    parser.add_argument("--xml_path", type=str,
                        default="./asset/example_scene_cube.xml")
    parser.add_argument("--delete-old", action="store_true",
                        help="root 已存在时直接删除，不再交互询问")
    # 接管触发参数
    parser.add_argument("--policy_max_steps", type=int, default=400,
                        help="策略阶段单回合最大步数（20Hz）")
    parser.add_argument("--grasp_deadline", type=int, default=120,
                        help="超过该步数仍未抬起立方体则接管")
    parser.add_argument("--hold_deadline", type=int, default=250,
                        help="夹持状态下超过该步数未完成放置则接管")
    parser.add_argument("--fsm_max_steps", type=int, default=1200,
                        help="FSM 恢复阶段最大步数（超过判负重试）")
    parser.add_argument("--max_retries", type=int, default=30,
                        help="单回合最大失败重试次数")
    args = parser.parse_args()

    seed = args.seed if args.seed >= 0 else None
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}, seed={seed}, num_demo={args.num_demo}, "
          f"root={args.root}, ckpt={args.ckpt}")

    # ------------------------------------------------------------- policy
    # 与 cube_deploy 完全一致的部署配置（temporal ensemble 0.9）
    dataset_metadata = LeRobotDatasetMetadata(args.repo_name, root=args.stats_root)
    features = dataset_to_policy_features(dataset_metadata.features)
    output_features = {k: ft for k, ft in features.items()
                       if ft.type is FeatureType.ACTION}
    input_features = {k: ft for k, ft in features.items()
                      if k not in output_features}
    with open(Path(args.ckpt) / "config.json") as f:
        ckpt_cfg = json.load(f)
    use_wrist = "observation.wrist_image" in ckpt_cfg["input_features"]
    if not use_wrist:
        input_features.pop("observation.wrist_image", None)

    cfg = ACTConfig(input_features=input_features,
                    output_features=output_features,
                    chunk_size=10, n_action_steps=1,
                    temporal_ensemble_coeff=0.9)
    resolve_delta_timestamps(cfg, dataset_metadata)
    policy = ACTPolicy.from_pretrained(args.ckpt, config=cfg,
                                       dataset_stats=dataset_metadata.stats)
    policy.to(device)
    policy.eval()

    # ------------------------------------------------------------ dataset
    dataset = build_dataset(args.repo_name, args.root, args.delete_old)
    n_episodes_before = len(dataset.meta.episodes)

    # --------------------------------------------------------------- env
    env = SimpleEnv(args.xml_path, action_type='joint_angle', seed=seed)
    checker = ContactGraspChecker(env.env, min_geoms=HOLD_MIN_GEOMS)
    fsm = CubePickPlaceFSM(env, checker)
    success_detector = PlaceSuccessDetector(checker)
    img_transform = torchvision.transforms.ToTensor()

    # ------------------------------------------------------------- utils
    def record_frame():
        """观测在动作施加前读取（与 cube_collect_data 一致）。"""
        ee_pose = env.get_ee_pose()
        agent_img, wrist_img = env.grab_image()
        agent_img = np.array(Image.fromarray(agent_img).resize((256, 256)))
        wrist_img = np.array(Image.fromarray(wrist_img).resize((256, 256)))
        return ee_pose, agent_img, wrist_img

    def add_frame(ee_pose, agent_img, wrist_img, action):
        """记录真实下发的关节目标 + 夹爪命令（与 cube_collect_data 一致）。"""
        recorded = np.concatenate(
            [env.compute_q.astype(np.float32),
             np.array([action[6]], dtype=np.float32)])
        dataset.add_frame({
            "observation.image": agent_img,
            "observation.wrist_image": wrist_img,
            "observation.state": ee_pose,
            "action": recorded,
            "obj_init": env.obj_init_pose,
        }, task=args.task_name)

    def takeover(held: bool, reason: str):
        """切换 action_type 并把 FSM 从当前状态切入。"""
        env.action_type = 'eef_pose'
        p, R = env.env.get_pR_body('tcp_link')
        env.p0, env.R0 = p.copy(), R.copy()
        fsm.begin_episode(already_held=held)
        print(f"[dagger] TAKEOVER step={step} reason='{reason}' "
              f"held={held} tcp=({p[0]:.2f},{p[1]:.2f},{p[2]:.2f})")

    def reset_episode():
        env.reset(seed=seed)
        wait_until_settled(env)
        policy.reset()
        success_detector.reset()
        env.action_type = 'joint_angle'
        return float(env.env.get_p_body(CUBE_BODY)[2])

    # ------------------------------------------------------------- loop
    n_saved = 0
    n_policy_only = 0
    n_takeover_saved = 0
    attempt = 0
    step = 0
    ever_lifted = False
    phase = 'policy'          # 'policy' -> 策略直播；'fsm' -> FSM 接管恢复
    fsm_steps = 0
    takeovers = []            # (step, reason) 每回合接管记录

    cube_init_z = reset_episode()
    print(f"dataset episodes before: {n_episodes_before}")

    while env.env.is_viewer_alive() and n_saved < args.num_demo:
        env.step_env()
        if not env.env.loop_every(HZ=20):
            continue

        if phase == 'policy':
            ee_pose, agent_img, wrist_img = record_frame()
            image = img_transform(Image.fromarray(agent_img))
            data = {
                'observation.state': torch.tensor([ee_pose]).to(device),
                'observation.image': image.unsqueeze(0).to(device),
                'task': [args.task_name],
                'timestamp': torch.tensor([step / 20.0]).to(device),
            }
            if use_wrist:
                wrist = img_transform(Image.fromarray(wrist_img))
                data['observation.wrist_image'] = wrist.unsqueeze(0).to(device)
            with torch.no_grad():
                action = policy.select_action(data)
            action = action[0].cpu().detach().numpy()
            env.step(action)
            add_frame(ee_pose, agent_img, wrist_img, action)
            env.render()
            step += 1

            cube_z = float(env.env.get_p_body(CUBE_BODY)[2])
            if not ever_lifted and cube_z > cube_init_z + LIFT_Z_MARGIN:
                ever_lifted = True

            # 策略独立成功 -> 直接保存（等价普通 demo）
            if success_detector.check(env):
                dataset.save_episode()
                n_saved += 1
                n_policy_only += 1
                print(f">>> Episode {n_saved}/{args.num_demo} saved "
                      f"[policy only, {step} frames]")
                step, ever_lifted, attempt = 0, False, 0
                cube_init_z = reset_episode()
                continue

            # 偏离检测 -> 接管
            held = is_holding(env, checker, cube_init_z)
            reason = None
            if cube_dropped(env, checker, cube_init_z, ever_lifted):
                reason = 'cube dropped'
            elif step >= args.grasp_deadline and not ever_lifted:
                reason = 'no grasp'
            elif step >= args.hold_deadline and held:
                reason = 'hold stall'
            elif step >= args.policy_max_steps:
                reason = 'policy timeout (held)' if held else \
                         'policy timeout (no grasp)'
            if reason:
                takeovers.append((step, reason))
                phase = 'fsm'
                fsm_steps = 0
                takeover(held, reason)
            continue

        # -------------------------------------------------- FSM 恢复阶段
        fsm_action, finished = fsm.step()
        ee_pose, agent_img, wrist_img = record_frame()
        env.step(fsm_action)
        add_frame(ee_pose, agent_img, wrist_img, fsm_action)
        env.render()
        step += 1
        fsm_steps += 1

        if finished and fsm.state == State.DONE:
            dataset.save_episode()
            n_saved += 1
            n_takeover_saved += 1
            print(f">>> Episode {n_saved}/{args.num_demo} saved "
                  f"[takeover: {takeovers}, {step} frames]")
            takeovers = []
            step, ever_lifted, attempt = 0, False, 0
            phase = 'policy'
            cube_init_z = reset_episode()
        elif finished or fsm_steps >= args.fsm_max_steps:
            attempt += 1
            if attempt > args.max_retries:
                print(f"Exceeded {args.max_retries} retries; aborting.")
                break
            print(f"Retry {attempt}/{args.max_retries} "
                  f"(reason: {fsm.failed_reason or 'fsm max steps'})")
            dataset.clear_episode_buffer()
            takeovers = []
            step, ever_lifted = 0, False
            phase = 'policy'
            cube_init_z = reset_episode()

    env.env.close_viewer()

    print(f"\nDone. collected {n_saved} dagger episodes into {args.root}")
    print(f"  policy-only : {n_policy_only}")
    print(f"  with takeover: {n_takeover_saved}")


if __name__ == "__main__":
    main()
