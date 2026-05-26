"""Evaluate the minimal RGCT-Dual v9 sharp few-shot classifier."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms

from easyfsl.datasets import CUB
from easyfsl.datasets.wrap_few_shot_dataset import WrapFewShotDataset
from easyfsl.samplers import TaskSampler
from easyfsl.utils import evaluate

import dino.vision_transformer as vits
from data.chestx import ChestX
from data.isic import ISICDataset
from experiments.methods.rgct.method import RGCTDualNet
from experiments.methods.rgct.variants import RGCT_DUAL_V9_SHARP_PARAMS
from utils.wandb import WandbLogger


DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
SEED = 42


def str2bool(value) -> bool:
    if isinstance(value, bool):
        return value
    value = str(value).strip().lower()
    if value in {"1", "true", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected boolean value, got {value!r}")


def set_seed(seed: int) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class MiniImageNetDataset(data.Dataset):
    """Load miniImageNet from a CrossDomainFewShot-style JSON filelist."""

    def __init__(self, json_path: str, transform=None):
        with open(json_path, "r") as f:
            meta = json.load(f)
        self.image_names = meta["image_names"]
        raw_labels = meta["image_labels"]
        unique = sorted(set(raw_labels))
        remap = {old: new for new, old in enumerate(unique)}
        self.targets = [remap[label] for label in raw_labels]
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_names)

    def __getitem__(self, index: int):
        image = Image.open(self.image_names[index]).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, self.targets[index]


class MyDataSet(data.Dataset):
    """Simple class-folder image dataset."""

    def __init__(self, root: str, transform=None):
        self.root = root
        self.transform = transform
        self.data, self.targets = self._load_samples()

    def _load_samples(self) -> Tuple[List[str], List[int]]:
        samples: List[str] = []
        targets: List[int] = []
        for class_idx, class_name in enumerate(sorted(os.listdir(self.root))):
            class_dir = os.path.join(self.root, class_name)
            if not os.path.isdir(class_dir):
                continue
            for sample_name in sorted(os.listdir(class_dir)):
                sample_path = os.path.join(class_dir, sample_name)
                if os.path.isfile(sample_path):
                    samples.append(sample_path)
                    targets.append(class_idx)
        return samples, targets

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int):
        image = Image.open(self.data[index]).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, self.targets[index]


class FastWrapFewShotDataset(WrapFewShotDataset):
    """WrapFewShotDataset variant that reuses precomputed labels when available."""

    def __init__(
        self,
        dataset,
        image_position_in_get_item_output: int = 0,
        label_position_in_get_item_output: int = 1,
        label_attribute_candidates: Tuple[str, ...] = ("targets", "labels", "y"),
    ):
        if image_position_in_get_item_output == label_position_in_get_item_output:
            raise ValueError("image and label positions must differ")

        item_length = len(dataset[0])
        if image_position_in_get_item_output >= item_length or label_position_in_get_item_output >= item_length:
            raise ValueError("specified output positions are out of range")

        labels_seq = None
        labels_source = None
        for attr in label_attribute_candidates:
            if hasattr(dataset, attr):
                candidate = getattr(dataset, attr)
                if callable(candidate):
                    candidate = candidate()
                if candidate is not None:
                    labels_seq = candidate
                    labels_source = attr
                    break

        if labels_seq is None:
            super().__init__(
                dataset,
                image_position_in_get_item_output,
                label_position_in_get_item_output,
            )
            return

        labels = self._materialize_labels(labels_seq)
        if len(labels) != len(dataset):
            super().__init__(
                dataset,
                image_position_in_get_item_output,
                label_position_in_get_item_output,
            )
            return

        self.source_dataset = dataset
        self.labels = labels
        self.image_position_in_get_item_output = image_position_in_get_item_output
        self.label_position_in_get_item_output = label_position_in_get_item_output
        self.labels_source = labels_source

    @staticmethod
    def _materialize_labels(labels_seq):
        if isinstance(labels_seq, torch.Tensor):
            labels = labels_seq.detach().cpu().tolist()
        elif isinstance(labels_seq, np.ndarray):
            labels = labels_seq.tolist()
        elif isinstance(labels_seq, list):
            labels = labels_seq
        else:
            labels = list(labels_seq)

        if labels and isinstance(labels[0], (np.generic, np.integer)):
            return [int(label) for label in labels]
        return labels


def build_ds_transforms(image_size: int) -> Dict[str, T.Compose]:
    imagenet = T.Compose(
        [
            transforms.PILToTensor(),
            transforms.Lambda(lambda x: x.float() / 255.0),
            transforms.Resize((image_size, image_size)),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return {
        "BCCD_WBC": T.Compose(
            [
                transforms.PILToTensor(),
                transforms.Lambda(lambda x: x.float() / 255.0),
                transforms.Resize((image_size, image_size)),
                T.Normalize(mean=[0.6659, 0.6028, 0.7932], std=[0.1221, 0.1698, 0.0543]),
            ]
        ),
        "Plant-Disease": T.Compose(
            [
                transforms.PILToTensor(),
                transforms.Lambda(lambda x: x.float() / 255.0),
                transforms.Resize((image_size, image_size)),
                T.Normalize(mean=[0.4662, 0.4888, 0.4101], std=[0.1707, 0.1438, 0.1875]),
            ]
        ),
        "EUROSAT": T.Compose(
            [
                transforms.PILToTensor(),
                transforms.Lambda(lambda x: x.float() / 255.0),
                transforms.Resize((image_size, image_size)),
                T.Normalize(mean=[0.3444, 0.3803, 0.4078], std=[0.0884, 0.0621, 0.0521]),
            ]
        ),
        "ChestX": T.Compose(
            [
                transforms.PILToTensor(),
                transforms.Lambda(lambda x: x.float() / 255.0),
                transforms.Resize((image_size, image_size)),
                T.Normalize(mean=[0.4920, 0.4920, 0.4920], std=[0.2288, 0.2288, 0.2288]),
            ]
        ),
        "ISIC": T.Compose(
            [
                transforms.PILToTensor(),
                transforms.Lambda(lambda x: x.float() / 255.0),
                transforms.Resize((image_size, image_size)),
                T.Normalize(mean=[0.7635, 0.5461, 0.5705], std=[0.0891, 0.1179, 0.1325]),
            ]
        ),
        "HEp": T.Compose(
            [
                T.ToTensor(),
                T.Normalize(mean=[0.7940, 0.7940, 0.7940], std=[0.1920, 0.1920, 0.1920]),
                T.Resize(size=(image_size, image_size)),
            ]
        ),
        "CUB": imagenet,
        "miniImageNet": imagenet,
        "tieredImageNet": imagenet,
    }


def load_dino_backbone(arch_name: str, image_size: int, device: str):
    if arch_name == "vits16":
        patch_size = 16
        url = "dino_deitsmall16_pretrain/dino_deitsmall16_pretrain.pth"
        arch = "vit_small"
    elif arch_name == "vits8":
        patch_size = 8
        url = "dino_deitsmall8_pretrain/dino_deitsmall8_pretrain.pth"
        arch = "vit_small"
    else:
        raise ValueError("backbone must be 'vits16' or 'vits8'")

    model = vits.__dict__[arch](patch_size=patch_size, num_classes=0)
    state_dict = torch.hub.load_state_dict_from_url("https://dl.fbaipublicfiles.com/dino/" + url)
    model.load_state_dict(state_dict, strict=True)
    model.to(device)

    default_transform = T.Compose(
        [
            transforms.PILToTensor(),
            transforms.Lambda(lambda x: x.float() / 255.0),
            transforms.Resize((image_size, image_size)),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return model, patch_size, default_transform


def build_dataset(args, transform):
    if args.dataset == "CUB":
        return CUB(split="test", training=False, transform=transform)
    if args.dataset == "Plant-Disease":
        return FastWrapFewShotDataset(
            MyDataSet(
                "/home/ahmedm04/projects/distill_part_whole/datasets/Plant-Disease/Plant-Disease",
                transform=transform,
            )
        )
    if args.dataset == "BCCD_WBC":
        return FastWrapFewShotDataset(
            MyDataSet(
                "/home/ahmedm04/projects/distill_part_whole/datasets/BCCD_WBC/BCCD_WBC",
                transform=transform,
            )
        )
    if args.dataset == "ChestX":
        return FastWrapFewShotDataset(ChestX("/home/ahmedm04/projects/DINOSEG/datasets/ChestX", transform=transform))
    if args.dataset == "ISIC":
        return FastWrapFewShotDataset(ISICDataset("/home/ahmedm04/projects/DINOSEG/datasets/ISIC2018", transform=transform))
    if args.dataset == "EUROSAT":
        return FastWrapFewShotDataset(
            MyDataSet("/home/ahmedm04/projects/distill_part_whole/datasets/EUROSAT/EUROSAT", transform=transform)
        )
    if args.dataset == "HEp":
        return FastWrapFewShotDataset(
            MyDataSet(
                "/home/ahmedm04/projects/distill_part_whole/datasets/HEp-Dataset/HEp-Dataset",
                transform=transform,
            )
        )
    if args.dataset == "miniImageNet":
        if args.mini_imagenet_root:
            return FastWrapFewShotDataset(MyDataSet(args.mini_imagenet_root, transform=transform))
        if args.mini_imagenet_json:
            return FastWrapFewShotDataset(MiniImageNetDataset(args.mini_imagenet_json, transform=transform))
        default_root = "/home_old/ahmedm04/mini-imagenet-tools/mini_imagenet_split/val"
        return FastWrapFewShotDataset(MyDataSet(default_root, transform=transform))
    if args.dataset == "tieredImageNet":
        root = args.tiered_imagenet_root or "/home_old/ahmedm04/tiered-imagenet-tools/tiered_imagenet/val"
        return FastWrapFewShotDataset(MyDataSet(root, transform=transform))
    raise ValueError(f"Unknown dataset: {args.dataset}")


def add_rgct_args(parser: argparse.ArgumentParser) -> None:
    defaults = RGCT_DUAL_V9_SHARP_PARAMS
    parser.add_argument("--reg_eps", type=float, default=defaults["reg_eps"])
    parser.add_argument("--reg_mass", type=float, default=defaults["reg_mass"])
    parser.add_argument("--sinkhorn_iters", type=int, default=defaults["sinkhorn_iters"])
    parser.add_argument("--use_ctb", type=str2bool, default=defaults["use_ctb"])
    parser.add_argument("--n_support_atoms", type=int, default=defaults["n_support_atoms"])
    parser.add_argument("--bary_iters", type=int, default=defaults["bary_iters"])
    parser.add_argument("--bary_inner_max", type=int, default=defaults["bary_inner_max"])
    parser.add_argument("--support_mix", type=float, default=defaults["support_mix"])
    parser.add_argument("--support_trim_ratio", type=float, default=defaults["support_trim_ratio"])
    parser.add_argument("--support_gate_temp", type=float, default=defaults["support_gate_temp"])
    parser.add_argument("--support_num_iter", type=int, default=defaults["support_num_iter"])
    parser.add_argument("--alpha_global", type=float, default=defaults["alpha_global"])
    parser.add_argument("--calibrate", type=str2bool, default=defaults["calibrate_episode"])
    parser.add_argument(
        "--episodic_trans_mode",
        type=str,
        default=defaults["episodic_trans_mode"],
        choices=["none", "support", "support_query", "query"],
    )
    parser.add_argument("--lambda_tv", type=float, default=defaults["lambda_tv"])
    parser.add_argument("--lambda_clutter", type=float, default=defaults["lambda_clutter"])
    parser.add_argument("--rgct_outer_iters", type=int, default=defaults["rgct_outer_iters"])
    parser.add_argument("--tau_z", type=float, default=defaults["tau_z"])
    parser.add_argument("--z_pdhg_iters", type=int, default=defaults["z_pdhg_iters"])
    parser.add_argument("--use_clutter", type=str2bool, default=defaults["use_clutter"])
    parser.add_argument("--anisotropic_tv", type=str2bool, default=defaults["anisotropic_tv"])
    parser.add_argument(
        "--rgct_scoring",
        type=str,
        default=defaults["rgct_scoring"],
        choices=["primal", "mass", "hybrid"],
    )
    parser.add_argument("--max_patches", type=int, default=0)


def collect_rgct_params(args) -> Dict[str, object]:
    return {
        "reg_eps": args.reg_eps,
        "reg_mass": args.reg_mass,
        "sinkhorn_iters": args.sinkhorn_iters,
        "use_ctb": args.use_ctb,
        "n_support_atoms": args.n_support_atoms,
        "bary_iters": args.bary_iters,
        "bary_inner_max": args.bary_inner_max,
        "support_mix": args.support_mix,
        "support_trim_ratio": args.support_trim_ratio,
        "support_gate_temp": args.support_gate_temp,
        "support_num_iter": args.support_num_iter,
        "alpha_global": args.alpha_global,
        "calibrate_episode": args.calibrate,
        "episodic_trans_mode": args.episodic_trans_mode,
        "scoring_mode": "rgct_dual",
        "lambda_tv": args.lambda_tv,
        "lambda_clutter": args.lambda_clutter,
        "rgct_outer_iters": args.rgct_outer_iters,
        "tau_z": args.tau_z,
        "z_pdhg_iters": args.z_pdhg_iters,
        "use_clutter": args.use_clutter,
        "anisotropic_tv": args.anisotropic_tv,
        "rgct_scoring": args.rgct_scoring,
        "max_patches": args.max_patches,
    }


def main() -> None:
    date_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    default_run_name = f"RGCTDual_v9_sharp_{date_time}"

    parser = argparse.ArgumentParser(description="RGCT-Dual v9 sharp few-shot evaluation")
    parser.add_argument("--n_way", type=int, default=5)
    parser.add_argument("--n_shot", type=int, default=5)
    parser.add_argument("--n_query", type=int, default=10)
    parser.add_argument("--n_test_tasks", type=int, default=100)
    parser.add_argument("--n_workers", type=int, default=12)
    parser.add_argument("--device", type=str, default=DEVICE)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--use_specific_trans", type=str2bool, default=False)
    parser.add_argument("--backbone", type=str, required=True, choices=["vits16", "vits8"])
    add_rgct_args(parser)

    parser.add_argument("--use_wandb", type=str2bool, default=True)
    parser.add_argument("--wandb_project", type=str, default="RGCT")
    parser.add_argument("--wandb_entity", type=str, default="leathead_AQ_AM_IO")
    parser.add_argument("--wandb_offline", type=str2bool, default=False)
    parser.add_argument("--wandb_name", type=str, default=default_run_name)
    parser.add_argument("--wandb_tags", type=str, default="")
    parser.add_argument(
        "--mini_imagenet_root",
        type=str,
        default="",
        help="Path to miniImageNet val/ folder with class subdirectories.",
    )
    parser.add_argument(
        "--mini_imagenet_json",
        type=str,
        default="",
        help="Path to miniImageNet JSON filelist. Used if --mini_imagenet_root is empty.",
    )
    parser.add_argument(
        "--tiered_imagenet_root",
        type=str,
        default="",
        help="Path to tieredImageNet split folder.",
    )
    parser.add_argument("--note", type=str, default="")
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    set_seed(args.seed)

    image_size = 224
    model, patch_size, default_transform = load_dino_backbone(args.backbone, image_size, args.device)
    transforms_by_dataset = build_ds_transforms(image_size)
    if args.use_specific_trans:
        if args.dataset not in transforms_by_dataset:
            raise ValueError(f"No dataset-specific transform registered for {args.dataset}")
        transform = transforms_by_dataset[args.dataset]
        print("LOADED DATASET-SPECIFIC TRANSFORMS")
    else:
        transform = default_transform

    test_set = build_dataset(args, transform)
    test_sampler = TaskSampler(
        test_set,
        n_way=args.n_way,
        n_shot=args.n_shot,
        n_query=args.n_query,
        n_tasks=args.n_test_tasks,
    )
    test_loader = DataLoader(
        test_set,
        batch_sampler=test_sampler,
        num_workers=args.n_workers,
        pin_memory=True,
        collate_fn=test_sampler.episodic_collate_fn,
    )

    rgct_params = collect_rgct_params(args)
    clf = RGCTDualNet(backbone=model, patch_size=patch_size, **rgct_params).to(args.device)

    episode_tag = f"{args.n_way}way{args.n_shot}shot"
    tags = [tag for tag in [args.dataset, args.backbone, episode_tag, "rgct_dual_v9_sharp"] if tag]
    if args.wandb_tags:
        tags.extend([tag.strip() for tag in args.wandb_tags.split(",") if tag.strip()])

    wandb_config = {
        "method": "rgct_dual_v9_sharp",
        "dataset": args.dataset,
        "backbone": args.backbone,
        "patch_size": patch_size,
        "image_size": image_size,
        "n_way": args.n_way,
        "n_shot": args.n_shot,
        "n_query": args.n_query,
        "n_test_tasks": args.n_test_tasks,
        "seed": args.seed,
        "device": args.device,
        "use_specific_trans": args.use_specific_trans,
        **rgct_params,
    }
    wb = WandbLogger(
        config=wandb_config,
        project=args.wandb_project,
        entity=args.wandb_entity,
        job_type="evaluation",
        offline=bool(args.wandb_offline),
        name=args.wandb_name,
        tags=tags,
        enabled=bool(args.use_wandb),
    )

    import wandb

    if wandb.run and not wandb.run.sweep_id:
        wb.log_config(wandb_config)

    acc = evaluate(clf, test_loader, device=args.device)
    print(f"Average accuracy : {(100.0 * acc):.2f} %")

    wb.log({"avg_accuracy": float(acc), "avg_accuracy_pct": float(round(100.0 * acc, 2))})
    wb.log_summary(
        {
            "avg_accuracy": float(round(acc, 2)),
            "avg_accuracy_pct": float(round(100.0 * acc, 2)),
        }
    )

    with open("logs.txt", "a+") as f:
        hparams = ", ".join(f"{key}: {value}" for key, value in sorted(rgct_params.items()))
        f.write(
            f"[RGCTDual_v9_sharp] Dataset: {args.dataset}, Backbone: {args.backbone}, "
            f"Nway: {args.n_way}, Nshot: {args.n_shot}, Nquery: {args.n_query}, "
            f"use_specific_trans: {args.use_specific_trans}, {hparams}, "
            f"Accuracy: {acc:.6f}, Note: {args.note}\n"
        )

    wb.finish_run()


if __name__ == "__main__":
    main()
