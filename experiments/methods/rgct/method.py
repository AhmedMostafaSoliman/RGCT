"""Minimal RGCT-Dual few-shot classifier.

This module intentionally contains only the implementation needed for
``rgct_dual_v9_sharp``.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from easyfsl.methods import FewShotClassifier

from utils.sinkhorn_unbalanced_torch import (
    sinkhorn_balanced_torch,
    sinkhorn_knopp_unbalanced_torch,
)


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _module_device(module: torch.nn.Module, fallback: torch.device) -> torch.device:
    try:
        return next(module.parameters()).device
    except StopIteration:
        return fallback


@torch.no_grad()
def vit_patch_tokens(backbone: torch.nn.Module, images: torch.Tensor) -> Tuple[torch.Tensor, int, int]:
    """Return L2-normalized ViT patch tokens as ``[B, P, D]``."""
    feat_all, _, _ = backbone.get_intermediate_feat(images, n=1)
    feat = feat_all[-1]
    batch, length, dim = feat.shape
    patches = length - 1
    ph = pw = int(math.sqrt(patches))
    if ph * pw != patches:
        raise ValueError(f"Expected square ViT patch grid, got {patches} patches")
    tokens = feat[:, 1:, :].reshape(batch, ph, pw, dim).permute(0, 3, 1, 2)
    tokens = tokens.flatten(2).transpose(1, 2)
    return F.normalize(tokens, dim=2, eps=1e-12), ph, pw


@torch.no_grad()
def vit_whole_embeddings(backbone: torch.nn.Module, images: torch.Tensor) -> torch.Tensor:
    """Return L2-normalized whole-image embeddings as ``[B, D]``."""
    return F.normalize(backbone(images), dim=1, eps=1e-12)


def unbalanced_barycenter_fixed_support(
    measures: List[torch.Tensor],
    n_support: int = 64,
    reg: float = 0.02,
    reg_m: float = 0.3,
    numItermax: int = 5,
    inner_max: int = 30,
    stopThr: float = 1e-4,
    measure_weights: Optional[List[torch.Tensor]] = None,
) -> torch.Tensor:
    """Fixed-support unbalanced Wasserstein barycenter used by CTB support."""
    if not measures:
        raise ValueError("measures must be non-empty")

    n_measures = len(measures)
    device = measures[0].device
    all_points = torch.cat(measures, dim=0)
    n_total = all_points.size(0)

    if n_total <= n_support:
        support = all_points.double()
    else:
        idx = torch.randperm(n_total, device=device)[:n_support]
        support = all_points[idx].double()

    a = torch.full(
        (support.size(0),),
        1.0 / float(support.size(0)),
        device=device,
        dtype=torch.float64,
    )

    b_list: List[torch.Tensor] = []
    if measure_weights is not None:
        if len(measure_weights) != n_measures:
            raise ValueError(
                f"measure_weights must have length {n_measures}, got {len(measure_weights)}"
            )
        for idx, (measure, weights) in enumerate(zip(measures, measure_weights)):
            if weights.numel() != measure.size(0):
                raise ValueError(
                    f"measure_weights[{idx}] has {weights.numel()} entries, "
                    f"expected {measure.size(0)}"
                )
            weights = weights.to(device=device, dtype=torch.float64).clamp(min=1e-12)
            b_list.append(weights / weights.sum().clamp(min=1e-12))
    else:
        for measure in measures:
            b_list.append(
                torch.full(
                    (measure.size(0),),
                    1.0 / float(measure.size(0)),
                    device=device,
                    dtype=torch.float64,
                )
            )

    for _ in range(int(numItermax)):
        numerator = torch.zeros_like(support)
        denominator = torch.zeros((support.size(0), 1), device=device, dtype=torch.float64)

        for measure, b in zip(measures, b_list):
            target = measure.double()
            sim = torch.clamp(support @ target.t(), -1.0, 1.0)
            cost = 1.0 - sim
            plan = sinkhorn_knopp_unbalanced_torch(
                M=cost.unsqueeze(0),
                a=a.unsqueeze(0),
                b=b.unsqueeze(0),
                reg=reg,
                reg_m=reg_m,
                numItermax=inner_max,
                stopThr=stopThr,
            )[0]
            numerator += plan @ target
            denominator += plan.sum(dim=1, keepdim=True)

        support = torch.where(denominator > 1e-12, numerator / denominator, support)
        support = F.normalize(support, p=2, dim=1, eps=1e-12)

    return support.float()


def _build_grid_edges(patches: int, device: torch.device) -> torch.Tensor:
    """Build directed 4-connected edges for a square patch grid."""
    height = width = int(math.sqrt(patches))
    if height * width != patches:
        height, width = 1, patches

    src: List[int] = []
    dst: List[int] = []
    for row in range(height):
        for col in range(width):
            idx = row * width + col
            if col + 1 < width:
                src.extend([idx, idx + 1])
                dst.extend([idx + 1, idx])
            if row + 1 < height:
                down = (row + 1) * width + col
                src.extend([idx, down])
                dst.extend([down, idx])
    return torch.tensor([src, dst], dtype=torch.long, device=device)


def _project_simplex(values: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Project each row of ``values`` onto the probability simplex."""
    rows, cols = values.shape
    sorted_values, _ = torch.sort(values, dim=1, descending=True)
    cumsum = torch.cumsum(sorted_values, dim=1)
    rho_range = torch.arange(1, cols + 1, device=values.device, dtype=values.dtype).unsqueeze(0)
    mask = (sorted_values - (cumsum - 1.0) / rho_range) > 0
    rho = cols - torch.flip(mask.int(), [1]).argmax(dim=1)
    rho = rho.clamp(min=1)
    theta = (cumsum[torch.arange(rows, device=values.device), rho - 1] - 1.0) / rho.to(values.dtype)
    projected = (values - theta.unsqueeze(1)).clamp(min=eps)
    return projected / projected.sum(dim=1, keepdim=True).clamp(min=eps)


def _apply_graph_gradient(values: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
    src, dst = edge_index[0], edge_index[1]
    return values[:, src] - values[:, dst]


def _apply_graph_divergence(values: torch.Tensor, edge_index: torch.Tensor, patches: int) -> torch.Tensor:
    batch, edges, channels = values.shape
    src, dst = edge_index[0], edge_index[1]
    div = torch.zeros((batch, patches, channels), device=values.device, dtype=values.dtype)
    div.scatter_add_(1, src.view(1, edges, 1).expand(batch, edges, channels), values)
    div.scatter_add_(1, dst.view(1, edges, 1).expand(batch, edges, channels), -values)
    return div


def _solve_semi_relaxed_ot_with_dual(
    cost: torch.Tensor,
    alpha: torch.Tensor,
    support_mass: torch.Tensor,
    reg: float,
    reg_mass_col: float,
    sinkhorn_iters: int,
    warmstart: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
) -> Tuple[torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """Solve semi-relaxed OT and return plan plus source dual potential."""
    plan, log_dict = sinkhorn_knopp_unbalanced_torch(
        M=cost,
        a=alpha,
        b=support_mass,
        reg=reg,
        reg_m=(float("inf"), reg_mass_col),
        numItermax=sinkhorn_iters,
        stopThr=1e-6,
        log=True,
        warmstart=warmstart,
    )
    logu = log_dict["logu"]
    logv = log_dict["logv"]
    return plan, reg * logu, (logu, logv)


def _rgct_z_step_pdhg(
    linear_cost: torch.Tensor,
    z_prev: torch.Tensor,
    edge_index: torch.Tensor,
    edge_weights: torch.Tensor,
    lambda_tv: float,
    tau_z: float,
    n_iters: int,
    eps: float = 1e-12,
) -> Tuple[torch.Tensor, List[float]]:
    """Solve the RGCT-Dual allocation subproblem via Chambolle-Pock PDHG."""
    batch, patches, channels = linear_cost.shape
    edges = edge_index.size(1)
    device = linear_cost.device
    dtype = linear_cost.dtype

    if edges > 0:
        degree = torch.zeros(patches, device=device, dtype=torch.long)
        degree.scatter_add_(0, edge_index[0], torch.ones(edges, device=device, dtype=torch.long))
        degree.scatter_add_(0, edge_index[1], torch.ones(edges, device=device, dtype=torch.long))
        graph_norm = max(2.0 * float(degree.max().item()), 1.0)
    else:
        graph_norm = 1.0

    sigma = 1.0 / (1.0 / tau_z + lambda_tv * graph_norm) if lambda_tv > 0 else tau_z
    tau_dual = 1.0 / (lambda_tv * graph_norm) if lambda_tv > 0 else 1.0

    z = z_prev.clone()
    z_bar = z.clone()
    dual = (
        torch.zeros((batch, edges, channels), device=device, dtype=dtype)
        if edges > 0 and lambda_tv > 0
        else None
    )
    obj_history: List[float] = []

    for _ in range(int(n_iters)):
        z_old = z.clone()

        if dual is not None:
            dual = dual + tau_dual * _apply_graph_gradient(z_bar, edge_index)
            threshold = lambda_tv * edge_weights.unsqueeze(2)
            dual = dual.clamp(-threshold, threshold)

        grad = linear_cost + (z - z_prev) / tau_z
        if dual is not None:
            grad = grad + _apply_graph_divergence(dual, edge_index, patches)

        z = z - sigma * grad
        for batch_idx in range(batch):
            z[batch_idx] = _project_simplex(z[batch_idx], eps=eps)

        z_bar = 2.0 * z - z_old

        obj_linear = (linear_cost * z).sum(dim=(1, 2))
        obj_prox = 0.5 / tau_z * ((z - z_prev) ** 2).sum(dim=(1, 2))
        if dual is not None:
            dz = _apply_graph_gradient(z, edge_index)
            obj_tv = (lambda_tv * edge_weights.unsqueeze(2) * dz.abs()).sum(dim=(1, 2))
        else:
            obj_tv = torch.zeros(batch, device=device, dtype=dtype)
        obj_history.append((obj_linear + obj_prox + obj_tv).mean().item())

    return z, obj_history


@torch.no_grad()
def rgct_dual_score_batched(
    query_tokens: torch.Tensor,
    support_classes: List[torch.Tensor],
    support_masses: List[torch.Tensor],
    reg_eps: float = 0.02,
    reg_mass: float = 0.3,
    sinkhorn_iters: int = 200,
    lambda_tv: float = 0.1,
    lambda_clutter: float = 0.5,
    outer_iters: int = 5,
    tau_z: float = 0.1,
    z_pdhg_iters: int = 50,
    use_clutter: bool = True,
    anisotropic_tv: bool = True,
    rgct_scoring: str = "primal",
    eps: float = 1e-12,
) -> torch.Tensor:
    """Compute RGCT-Dual class logits for a batch of query token sets."""
    batch, patches, _ = query_tokens.shape
    n_way = len(support_classes)
    device = query_tokens.device
    dtype = torch.float64

    n_channels = n_way + 1 if use_clutter else n_way
    class_offset = 1 if use_clutter else 0
    query_mass = torch.full((batch, patches), 1.0 / float(patches), device=device, dtype=dtype)

    costs: List[torch.Tensor] = []
    masses: List[torch.Tensor] = []
    for support, support_mass in zip(support_classes, support_masses):
        sim = torch.clamp(query_tokens.to(dtype) @ support.to(dtype).t(), -1.0, 1.0)
        cost = 1.0 - sim
        mass = support_mass.to(device=device, dtype=dtype).unsqueeze(0).expand(batch, -1)
        mass = mass * (query_mass.sum(dim=1, keepdim=True) / mass.sum(dim=1, keepdim=True).clamp(min=eps))
        costs.append(cost)
        masses.append(mass)

    edge_index = _build_grid_edges(patches, device)
    if anisotropic_tv and edge_index.size(1) > 0:
        q64 = query_tokens.to(dtype)
        weights = []
        for batch_idx in range(batch):
            src = q64[batch_idx, edge_index[0]]
            dst = q64[batch_idx, edge_index[1]]
            dist = ((src - dst) ** 2).sum(dim=1)
            median = dist.median().clamp(min=eps)
            weights.append(torch.exp(-dist / (2.0 * median)))
        edge_weights = torch.stack(weights)
    else:
        edge_weights = torch.ones((batch, edge_index.size(1)), device=device, dtype=dtype)

    z = torch.full((batch, patches, n_channels), 1.0 / float(n_channels), device=device, dtype=dtype)
    warmstarts: List[Optional[Tuple[torch.Tensor, torch.Tensor]]] = [None] * n_way
    plans: List[Optional[torch.Tensor]] = [None] * n_way
    duals: List[Optional[torch.Tensor]] = [None] * n_way

    for _ in range(int(outer_iters)):
        for class_idx in range(n_way):
            z_class = z[:, :, class_idx + class_offset].clamp(min=1e-8)
            alpha = query_mass * z_class
            plan, dual, warmstart = _solve_semi_relaxed_ot_with_dual(
                cost=costs[class_idx],
                alpha=alpha,
                support_mass=masses[class_idx],
                reg=reg_eps,
                reg_mass_col=reg_mass,
                sinkhorn_iters=sinkhorn_iters,
                warmstart=warmstarts[class_idx],
            )
            plans[class_idx] = plan
            duals[class_idx] = dual
            warmstarts[class_idx] = warmstart

        linear = torch.zeros((batch, patches, n_channels), device=device, dtype=dtype)
        for class_idx, dual in enumerate(duals):
            linear[:, :, class_idx + class_offset] = query_mass * dual
        if use_clutter:
            linear[:, :, 0] = linear[:, :, 0] + lambda_clutter

        z, _ = _rgct_z_step_pdhg(
            linear_cost=linear,
            z_prev=z,
            edge_index=edge_index,
            edge_weights=edge_weights,
            lambda_tv=lambda_tv,
            tau_z=tau_z,
            n_iters=z_pdhg_iters,
            eps=eps,
        )

    for class_idx in range(n_way):
        z_class = z[:, :, class_idx + class_offset].clamp(min=1e-8)
        alpha = query_mass * z_class
        plans[class_idx], _, _ = _solve_semi_relaxed_ot_with_dual(
            cost=costs[class_idx],
            alpha=alpha,
            support_mass=masses[class_idx],
            reg=reg_eps,
            reg_mass_col=reg_mass,
            sinkhorn_iters=sinkhorn_iters,
            warmstart=warmstarts[class_idx],
        )

    primal_costs = torch.zeros((batch, n_way), device=device, dtype=dtype)
    class_masses = torch.zeros((batch, n_way), device=device, dtype=dtype)
    for class_idx, plan in enumerate(plans):
        primal_costs[:, class_idx] = (plan * costs[class_idx]).sum(dim=(1, 2))
        class_masses[:, class_idx] = plan.sum(dim=(1, 2))

    scoring = str(rgct_scoring).strip().lower()
    if scoring == "primal":
        logits = -primal_costs
    elif scoring == "mass":
        logits = torch.log(class_masses.clamp(min=eps))
    elif scoring == "hybrid":
        logits = torch.log(class_masses.clamp(min=eps))
        logits = logits - 0.5 * primal_costs / primal_costs.abs().mean(dim=1, keepdim=True).clamp(min=eps)
    else:
        raise ValueError("rgct_scoring must be one of: primal, mass, hybrid")

    return logits.to(torch.float32)


def _calibrate_episode_logits(
    logits: torch.Tensor,
    n_way: int,
    reg_eps: float,
    eps: float = 1e-12,
) -> torch.Tensor:
    batch = logits.size(0)
    cost = -logits.to(torch.float64)
    query_mass = torch.full((batch,), 1.0 / float(batch), device=logits.device, dtype=torch.float64)
    class_mass = torch.full((n_way,), 1.0 / float(n_way), device=logits.device, dtype=torch.float64)
    plan = sinkhorn_balanced_torch(
        cost,
        query_mass,
        class_mass,
        reg=float(reg_eps),
        numItermax=500,
        stopThr=1e-6,
    ).to(torch.float32)
    return logits + torch.log(plan.clamp(min=eps))


class RGCTDualNet(FewShotClassifier):
    """EasyFSL classifier for the single ``rgct_dual_v9_sharp`` method."""

    def __init__(
        self,
        backbone: torch.nn.Module,
        patch_size: int = 16,
        reg_eps: float = 0.02,
        reg_mass: float = 0.3,
        sinkhorn_iters: int = 200,
        use_ctb: bool = True,
        n_support_atoms: int = 64,
        bary_iters: int = 5,
        bary_inner_max: int = 30,
        support_mix: float = 0.5,
        support_trim_ratio: float = 0.95,
        support_gate_temp: float = 0.5,
        support_num_iter: int = 120,
        alpha_global: float = 0.4,
        calibrate_episode: bool = True,
        episodic_trans_mode: str = "support",
        lambda_tv: float = 0.1,
        lambda_clutter: float = 0.5,
        rgct_outer_iters: int = 5,
        tau_z: float = 0.1,
        z_pdhg_iters: int = 50,
        use_clutter: bool = True,
        anisotropic_tv: bool = True,
        rgct_scoring: str = "primal",
        scoring_mode: str = "rgct_dual",
        max_patches: int = 0,
        eps: float = 1e-12,
    ) -> None:
        super().__init__()
        if str(scoring_mode).strip().lower() != "rgct_dual":
            raise ValueError("Only scoring_mode='rgct_dual' is supported in this module.")

        self.backbone = backbone
        self.patch_size = int(patch_size)
        self.reg_eps = float(reg_eps)
        self.reg_mass = float(reg_mass)
        self.sinkhorn_iters = int(sinkhorn_iters)
        self.use_ctb = bool(use_ctb)
        self.n_support_atoms = int(n_support_atoms)
        self.bary_iters = int(bary_iters)
        self.bary_inner_max = int(bary_inner_max)
        self.support_mix = float(support_mix)
        self.support_trim_ratio = float(support_trim_ratio)
        self.support_gate_temp = float(support_gate_temp)
        self.support_num_iter = int(support_num_iter)
        self.alpha_global = float(alpha_global)
        self.calibrate_episode = bool(calibrate_episode)
        self.episodic_trans_mode = str(episodic_trans_mode).strip().lower()
        self.lambda_tv = float(lambda_tv)
        self.lambda_clutter = float(lambda_clutter)
        self.rgct_outer_iters = int(rgct_outer_iters)
        self.tau_z = float(tau_z)
        self.z_pdhg_iters = int(z_pdhg_iters)
        self.use_clutter = bool(use_clutter)
        self.anisotropic_tv = bool(anisotropic_tv)
        self.rgct_scoring = str(rgct_scoring).strip().lower()
        self.max_patches = int(max_patches)
        self.eps = float(eps)

        if self.episodic_trans_mode not in {"none", "support", "support_query", "query"}:
            raise ValueError("episodic_trans_mode must be one of: none, support, support_query, query")

        self.support_images_cached: Optional[torch.Tensor] = None
        self.support_labels_cached: Optional[torch.Tensor] = None
        self.support_classes: List[torch.Tensor] = []
        self.support_masses: List[torch.Tensor] = []
        self.global_prototypes: Optional[torch.Tensor] = None
        self.n_way = 0

    def _episode_stats(self, images: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mean = images.mean(dim=(0, 2, 3), keepdim=True)
        std = images.std(dim=(0, 2, 3), keepdim=True, unbiased=False).clamp_min(1e-6)
        return mean, std

    def _resolve_images(self, query_images: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        support = self.support_images_cached
        if support is None:
            raise RuntimeError("process_support_set must be called before forward")

        mode = self.episodic_trans_mode
        if mode == "none":
            return support, query_images
        if mode == "support":
            source = support
        elif mode == "support_query":
            source = torch.cat([support, query_images], dim=0)
        else:
            source = query_images
        mean, std = self._episode_stats(source)
        return (support - mean) / std, (query_images - mean) / std

    def _support_patch_weights(self, shots: torch.Tensor, shot_idx: int) -> torch.Tensor:
        n_shots, n_patches, _ = shots.shape
        uniform = torch.full(
            (n_patches,),
            1.0 / float(n_patches),
            device=shots.device,
            dtype=torch.float64,
        )
        if n_shots <= 1:
            return uniform

        query = shots[shot_idx : shot_idx + 1]
        others = torch.cat([shots[idx] for idx in range(n_shots) if idx != shot_idx], dim=0)
        cost = (1.0 - torch.clamp(query @ others.t(), -1.0, 1.0)).to(dtype=torch.float64)
        target = torch.full(
            (1, others.size(0)),
            1.0 / float(others.size(0)),
            device=shots.device,
            dtype=torch.float64,
        )
        plan = sinkhorn_knopp_unbalanced_torch(
            M=cost,
            a=uniform.unsqueeze(0),
            b=target,
            reg=self.reg_eps,
            reg_m=self.reg_mass,
            numItermax=self.support_num_iter,
            stopThr=1e-6,
        )
        row = plan.sum(dim=2).squeeze(0).clamp(min=self.eps)
        row = row / row.sum().clamp(min=self.eps)
        z = (row - row.mean()) / row.std(unbiased=False).clamp(min=self.eps)
        gate = torch.sigmoid(z / max(self.support_gate_temp, 1e-3)).clamp(min=self.eps)
        gate = gate / gate.sum().clamp(min=self.eps)
        mix = float(np.clip(self.support_mix, 0.0, 1.0))
        weights = (1.0 - mix) * uniform + mix * gate
        return weights / weights.sum().clamp(min=self.eps)

    def _build_ctb_support(self, support_tokens: torch.Tensor) -> None:
        labels = self.support_labels_cached
        if labels is None:
            raise RuntimeError("support labels are not available")

        self.support_classes = []
        self.support_masses = []
        trim_ratio = float(np.clip(self.support_trim_ratio, 0.05, 1.0))
        classes = torch.unique(labels).tolist()
        self.n_way = len(classes)

        for class_id in classes:
            idx = torch.nonzero(labels == class_id, as_tuple=False).squeeze(1)
            shots = support_tokens[idx].contiguous()
            measures: List[torch.Tensor] = []
            weights: List[torch.Tensor] = []
            for shot_idx in range(shots.size(0)):
                shot = shots[shot_idx]
                weight = self._support_patch_weights(shots, shot_idx)
                keep = max(1, int(math.ceil(trim_ratio * shot.size(0))))
                if keep < shot.size(0):
                    top_idx = torch.topk(weight, k=keep, largest=True).indices
                    top_idx = torch.sort(top_idx).values
                    shot = shot[top_idx]
                    weight = weight[top_idx]
                weight = weight / weight.sum().clamp(min=self.eps)
                measures.append(shot.double())
                weights.append(weight.to(dtype=torch.float64))

            proto = unbalanced_barycenter_fixed_support(
                measures=measures,
                n_support=self.n_support_atoms,
                reg=self.reg_eps,
                reg_m=self.reg_mass,
                numItermax=self.bary_iters,
                inner_max=self.bary_inner_max,
                measure_weights=weights,
            )
            proto = proto.to(support_tokens.device)
            self.support_classes.append(proto)
            self.support_masses.append(
                torch.full(
                    (proto.size(0),),
                    1.0 / float(proto.size(0)),
                    device=support_tokens.device,
                    dtype=torch.float32,
                )
            )

    def _build_raw_support(self, support_tokens: torch.Tensor) -> None:
        labels = self.support_labels_cached
        if labels is None:
            raise RuntimeError("support labels are not available")

        self.support_classes = []
        self.support_masses = []
        classes = torch.unique(labels).tolist()
        self.n_way = len(classes)
        for class_id in classes:
            idx = torch.nonzero(labels == class_id, as_tuple=False).squeeze(1)
            support = support_tokens[idx].reshape(-1, support_tokens.size(-1)).float()
            self.support_classes.append(support)
            self.support_masses.append(
                torch.full(
                    (support.size(0),),
                    1.0 / float(support.size(0)),
                    device=support_tokens.device,
                    dtype=torch.float32,
                )
            )

    @torch.no_grad()
    def process_support_set(self, support_images: torch.Tensor, support_labels: torch.Tensor) -> None:
        fallback = torch.device(support_images.device)
        device = _module_device(self.backbone, fallback)
        self.support_images_cached = support_images.to(device)
        self.support_labels_cached = support_labels.to(device)
        self.support_classes = []
        self.support_masses = []
        self.global_prototypes = None
        self.n_way = int(torch.unique(self.support_labels_cached).numel())

    @torch.no_grad()
    def forward(self, query_images: torch.Tensor) -> torch.Tensor:
        if self.support_images_cached is None or self.support_labels_cached is None:
            raise RuntimeError("process_support_set must be called before forward")

        device = self.support_images_cached.device
        query_images = query_images.to(device)
        support_images, query_images = self._resolve_images(query_images)

        support_tokens, _, _ = vit_patch_tokens(self.backbone, support_images)
        query_tokens, _, _ = vit_patch_tokens(self.backbone, query_images)
        support_whole = vit_whole_embeddings(self.backbone, support_images)
        query_whole = vit_whole_embeddings(self.backbone, query_images)

        if self.max_patches and self.max_patches < support_tokens.size(1):
            stride = max(1, int(math.floor(support_tokens.size(1) / self.max_patches)))
            idx = torch.arange(0, support_tokens.size(1), stride, device=device)[: self.max_patches]
            support_tokens = support_tokens[:, idx]
        if self.max_patches and self.max_patches < query_tokens.size(1):
            stride = max(1, int(math.floor(query_tokens.size(1) / self.max_patches)))
            idx = torch.arange(0, query_tokens.size(1), stride, device=device)[: self.max_patches]
            query_tokens = query_tokens[:, idx]

        if self.use_ctb:
            self._build_ctb_support(support_tokens)
        else:
            self._build_raw_support(support_tokens)

        parts_logits = rgct_dual_score_batched(
            query_tokens=query_tokens,
            support_classes=self.support_classes,
            support_masses=self.support_masses,
            reg_eps=self.reg_eps,
            reg_mass=self.reg_mass,
            sinkhorn_iters=self.sinkhorn_iters,
            lambda_tv=self.lambda_tv,
            lambda_clutter=self.lambda_clutter,
            outer_iters=self.rgct_outer_iters,
            tau_z=self.tau_z,
            z_pdhg_iters=self.z_pdhg_iters,
            use_clutter=self.use_clutter,
            anisotropic_tv=self.anisotropic_tv,
            rgct_scoring=self.rgct_scoring,
            eps=self.eps,
        )

        centers = []
        for class_id in torch.unique(self.support_labels_cached).tolist():
            idx = torch.nonzero(self.support_labels_cached == class_id, as_tuple=False).squeeze(1)
            centers.append(F.normalize(support_whole[idx].mean(dim=0, keepdim=True), dim=1).squeeze(0))
        self.global_prototypes = torch.stack(centers, dim=0).to(device)

        global_logits = query_whole @ self.global_prototypes.t()
        alpha = float(np.clip(self.alpha_global, 0.0, 1.0))
        logits = (alpha * global_logits + (1.0 - alpha) * parts_logits).to(torch.float32)

        if self.calibrate_episode:
            logits = _calibrate_episode_logits(
                logits=logits,
                n_way=self.n_way,
                reg_eps=self.reg_eps,
                eps=self.eps,
            )

        return self.softmax_if_specified(logits)


__all__ = [
    "RGCTDualNet",
    "rgct_dual_score_batched",
    "unbalanced_barycenter_fixed_support",
    "vit_patch_tokens",
    "vit_whole_embeddings",
]
