#!/usr/bin/env python
"""在 cube 演示数据集上训练 ACT 策略（含 episode 级验证集）。

迁移自根目录 3.train.py，改动：
  1. 数据集/仓库名/输出目录指向 cube_pnp（./demo_data, ./ckpt/act_cube）。
  2. 新增 10% episode 级验证集（最后若干 episode 不参与训练），训练中定期
     输出 train/val loss，便于监控过拟合。
  3. 训练结束做 checkpoint 保存与重新加载校验，并在一集上打印预测动作
     与 GT 的平均误差。

Usage:
    python cube_train.py                          # 1500 steps, GPU, 10% val
    python cube_train.py --steps 200 --no-cuda    # CPU sanity check
"""
import os
import sys
import argparse
from pathlib import Path

# 相对路径以 cube_pnp/ 为准，mujoco_env 等仍从项目根目录导入
os.chdir(Path(__file__).resolve().parent)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
from torchvision import transforms
from lerobot.common.datasets.lerobot_dataset import (
    LeRobotDataset, LeRobotDatasetMetadata)
from lerobot.common.datasets.utils import dataset_to_policy_features
from lerobot.common.policies.act.configuration_act import ACTConfig
from lerobot.common.policies.act.modeling_act import ACTPolicy
from lerobot.configs.types import FeatureType
from lerobot.common.datasets.factory import resolve_delta_timestamps


# ===========================================================================
# 图像增强（与原脚本一致：高斯噪声 + clamp）
# ===========================================================================
class AddGaussianNoise:
    def __init__(self, mean=0.0, std=0.01):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean


# ===========================================================================
# Episode 级采样器：把 episode 切成 train / val 两份
# ===========================================================================
class EpisodeFrameSampler(torch.utils.data.Sampler):
    """按 episode_index 列表枚举全部帧；shuffle=True 时每轮打散。"""

    def __init__(self, dataset: LeRobotDataset, episode_indices, shuffle=False,
                 seed: int = 0):
        self.frame_ids = []
        for ep in episode_indices:
            lo = dataset.episode_data_index["from"][ep].item()
            hi = dataset.episode_data_index["to"][ep].item()
            self.frame_ids.extend(range(lo, hi))
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0

    def __iter__(self):
        ids = self.frame_ids
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            ids = [ids[i] for i in torch.randperm(len(ids), generator=g).tolist()]
            self.epoch += 1
        return iter(ids)

    def __len__(self):
        return len(self.frame_ids)


@torch.no_grad()
def evaluate_loss(policy, dataloader, device, max_batches=None):
    """返回验证集平均 loss。"""
    policy.eval()
    losses, n = 0.0, 0
    for bi, batch in enumerate(dataloader):
        if max_batches is not None and bi >= max_batches:
            break
        inp = {k: (v.to(device) if isinstance(v, torch.Tensor) else v)
               for k, v in batch.items()}
        loss, _ = policy.forward(inp)
        bs = inp["action"].shape[0]
        losses += loss.item() * bs
        n += bs
    policy.train()
    return losses / max(n, 1)


# ===========================================================================
# Main
# ===========================================================================
def main():
    parser = argparse.ArgumentParser(description="Train ACT on cube demos (with val split)")
    parser.add_argument("--repo_name", type=str, default="omy_cube_pnp")
    parser.add_argument("--root", type=str, default="./demo_data")
    parser.add_argument("--ckpt", type=str, default="./ckpt/act_cube")
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--val_ratio", type=float, default=0.1,
                        help="验证集占总 episode 的比例（按 episode 划分）")
    parser.add_argument("--log_freq", type=int, default=100)
    parser.add_argument("--val_freq", type=int, default=500)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-cuda", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda
                          else "cpu")
    print(f"device={device}")

    # ------------------------------------------------------------------ data
    dataset_metadata = LeRobotDatasetMetadata(args.repo_name, root=args.root)
    n_episodes = int(dataset_metadata.total_episodes)
    n_val = max(1, round(n_episodes * args.val_ratio))
    val_eps = list(range(n_episodes - n_val, n_episodes))
    train_eps = list(range(0, n_episodes - n_val))
    print(f"episodes total={n_episodes} train={train_eps} val={val_eps}")

    features = dataset_to_policy_features(dataset_metadata.features)
    output_features = {k: ft for k, ft in features.items()
                       if ft.type is FeatureType.ACTION}
    input_features = {k: ft for k, ft in features.items()
                      if k not in output_features}
    # 与原脚本一致：仅用 agentview 主视角，不用 wrist 视角
    input_features.pop("observation.wrist_image", None)

    cfg = ACTConfig(input_features=input_features,
                    output_features=output_features,
                    chunk_size=10, n_action_steps=10)
    delta_timestamps = resolve_delta_timestamps(cfg, dataset_metadata)

    transform = transforms.Compose([
        AddGaussianNoise(mean=0.0, std=0.02),
        transforms.Lambda(lambda x: x.clamp(0, 1)),
    ])
    dataset = LeRobotDataset(args.repo_name,
                             delta_timestamps=delta_timestamps,
                             root=args.root,
                             image_transforms=transform)

    train_sampler = EpisodeFrameSampler(dataset, train_eps, shuffle=True,
                                        seed=args.seed)
    val_sampler = EpisodeFrameSampler(dataset, val_eps, shuffle=False)
    train_loader = torch.utils.data.DataLoader(
        dataset, num_workers=args.num_workers, batch_size=args.batch_size,
        sampler=train_sampler, pin_memory=device.type != "cpu", drop_last=True)
    val_loader = torch.utils.data.DataLoader(
        dataset, num_workers=args.num_workers, batch_size=args.batch_size,
        sampler=val_sampler, pin_memory=device.type != "cpu")
    print(f"train frames={len(train_sampler)} val frames={len(val_sampler)}")

    # ---------------------------------------------------------------- model
    policy = ACTPolicy(cfg, dataset_stats=dataset_metadata.stats)
    policy.train()
    policy.to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=args.lr)

    # -------------------------------------------------------------- training
    step = 0
    done = False
    while not done:
        for batch in train_loader:
            inp = {k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                   for k, v in batch.items()}
            loss, _ = policy.forward(inp)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if step % args.log_freq == 0:
                print(f"step {step:5d}/{args.steps}  train loss {loss.item():.4f}",
                      flush=True)
            if args.val_freq > 0 and step > 0 and step % args.val_freq == 0:
                val_loss = evaluate_loss(policy, val_loader, device)
                print(f"step {step:5d}  >>> val loss {val_loss:.4f}", flush=True)
            step += 1
            if step >= args.steps:
                done = True
                break

    val_loss_final = evaluate_loss(policy, val_loader, device)
    print(f"FINAL train loss {loss.item():.4f}  val loss {val_loss_final:.4f}")

    # ------------------------------------------------------------- save ckpt
    ckpt_dir = Path(args.ckpt)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    policy.save_pretrained(str(ckpt_dir))
    print(f"checkpoint saved to {ckpt_dir}")

    # ----------------------------------------------------- ckpt load 验证
    reloaded = ACTPolicy.from_pretrained(str(ckpt_dir))
    reloaded.to(device)
    reloaded.eval()
    print("checkpoint reload: OK")

    # 一集上对比预测动作与 GT 的平均绝对误差
    one_ep_loader = torch.utils.data.DataLoader(
        dataset, num_workers=args.num_workers, batch_size=1,
        sampler=EpisodeFrameSampler(dataset, [train_eps[0]]),
        pin_memory=device.type != "cpu")
    errs = []
    reloaded.reset()
    with torch.no_grad():
        for batch in one_ep_loader:
            inp = {k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                   for k, v in batch.items()}
            pred = reloaded.select_action(inp)
            gt = inp["action"][:, 0, :]
            errs.append(torch.mean(torch.abs(pred - gt)).item())
    print(f"ep{train_eps[0]} mean |pred - gt| action error: "
          f"{np.mean(errs):.4f} rad (over {len(errs)} frames)")
    print("DONE")


if __name__ == "__main__":
    main()
