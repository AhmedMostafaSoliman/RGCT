"""Minimal RGCT-Dual few-shot classifier.

This module intentionally contains only the implementation needed for
``rgct_dual_v9_sharp``.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from easyfsl.methods import FewShotClassifier

from utils.sinkhorn_unbalanced_torch import (
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
    measures: List[torch.Tensor],   # List of [N_k, D] tensors
    n_support: int = 64,            # L: number of atoms in barycenter
    reg: float = 0.05,              # Entropy
    reg_m: float = 0.5,             # Marginal relaxation (unbalancedness)
    numItermax: int = 10,           # Outer barycenter iterations
    inner_max: int = 100,           # Inner Sinkhorn iterations
    stopThr: float = 1e-4,
    init_type: str = "random",      # 'random' or 'kmeans'
    measure_weights: Optional[List[torch.Tensor]] = None,  # Optional per-measure marginals
) -> torch.Tensor:
    """
    Computes the Unbalanced Wasserstein Barycenter with fixed support size.
    Returns: Z [n_support, D] (the prototype features)
    """
    K = len(measures)
    D = measures[0].shape[1]
    device = measures[0].device
    
    # 1. Initialize Z
    all_points = torch.cat(measures, dim=0)
    N_total = all_points.shape[0]
    
    if init_type == "random" or N_total <= n_support:
        if N_total <= n_support:
            Z = all_points.double()
        else:
            idx = torch.randperm(N_total, device=device)[:n_support]
            Z = all_points[idx].double()
    else:
        # Simple random default
        idx = torch.randperm(N_total, device=device)[:n_support]
        Z = all_points[idx].double()
        
    a = torch.full((Z.shape[0],), 1.0 / Z.shape[0], device=device, dtype=torch.float64)
    
    # Optional per-measure marginals (used for trimmed/weighted barycenters).
    b_list: List[torch.Tensor] = []
    if measure_weights is not None:
        if len(measure_weights) != K:
            raise ValueError(
                f"measure_weights must have length {K}, got {len(measure_weights)}"
            )
        for k in range(K):
            S_k = measures[k]
            b_k = measure_weights[k]
            if b_k.numel() != S_k.shape[0]:
                raise ValueError(
                    f"measure_weights[{k}] has {b_k.numel()} entries, expected {S_k.shape[0]}"
                )
            b_k = b_k.to(device=device, dtype=torch.float64).clamp(min=1e-12)
            b_k = b_k / (b_k.sum() + 1e-12)
            b_list.append(b_k)
    else:
        for k in range(K):
            N_k = measures[k].shape[0]
            b_k = torch.full((N_k,), 1.0 / N_k, device=device, dtype=torch.float64)
            b_list.append(b_k)

    # 2. Alternating Minimization
    for it in range(numItermax):
        numerator = torch.zeros_like(Z)
        denominator = torch.zeros((Z.shape[0], 1), device=device, dtype=torch.float64)
        
        for k in range(K):
            S_k = measures[k].double()
            b_k = b_list[k]
            
            sim = torch.clamp(Z @ S_k.t(), -1.0, 1.0)
            M_k = 1.0 - sim
            
            Gamma_k = sinkhorn_knopp_unbalanced_torch(
                M=M_k.unsqueeze(0), a=a.unsqueeze(0), b=b_k.unsqueeze(0),
                reg=reg, reg_m=reg_m, numItermax=inner_max, stopThr=stopThr
            )[0]
            
            numerator += Gamma_k @ S_k
            denominator += Gamma_k.sum(dim=1, keepdim=True)
            
        # Update Z
        valid_mask = (denominator > 1e-12)
        Z_new = torch.where(valid_mask, numerator / denominator, Z)
        
        # Normalize Z to unit sphere (Crucial for Cosine Distance / DINO)
        Z_new = torch.nn.functional.normalize(Z_new, p=2, dim=1)
        
        Z = Z_new
        
    return Z.float()


def _build_grid_edges(P: int, device: torch.device) -> torch.Tensor:
    """Build 4-connected grid edges for a square patch grid.

    Returns edge_index [2, E] in COO format.
    """
    H = W = int(math.sqrt(P))
    if H * W != P:
        # Fallback: chain graph
        H, W = 1, P
    src, dst = [], []
    for r in range(H):
        for c in range(W):
            idx = r * W + c
            if c + 1 < W:
                src.append(idx); dst.append(idx + 1)
                src.append(idx + 1); dst.append(idx)
            if r + 1 < H:
                src.append(idx); dst.append((r + 1) * W + c)
                src.append((r + 1) * W + c); dst.append(idx)
    return torch.tensor([src, dst], dtype=torch.long, device=device)


def _project_simplex(Z: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Project each row of Z onto the probability simplex Δ^K.

    Uses the efficient sorting-based algorithm of Condat (2016).
    Z: [P, K]  →  Z_proj: [P, K], each row sums to 1, all entries ≥ 0.
    """
    P, K = Z.shape
    sorted_z, _ = torch.sort(Z, dim=1, descending=True)
    cumsum = torch.cumsum(sorted_z, dim=1)
    rho_range = torch.arange(1, K + 1, device=Z.device, dtype=Z.dtype).unsqueeze(0)
    mask = (sorted_z - (cumsum - 1.0) / rho_range) > 0
    rho = K - torch.flip(mask.int(), [1]).argmax(dim=1)  # [P]
    rho = rho.clamp(min=1)
    theta = (cumsum[torch.arange(P, device=Z.device), rho - 1] - 1.0) / rho.to(Z.dtype)
    return (Z - theta.unsqueeze(1)).clamp(min=eps)


def _apply_graph_gradient(
    Z: torch.Tensor,
    edge_index: torch.Tensor,
) -> torch.Tensor:
    """Compute DZ: differences along graph edges.

    Z: [B, P, K]  →  DZ: [B, E, K]  where DZ[e] = Z[src[e]] - Z[dst[e]]
    """
    src, dst = edge_index[0], edge_index[1]
    return Z[:, src] - Z[:, dst]  # [B, E, K]


def _apply_graph_divergence(
    Y: torch.Tensor,
    edge_index: torch.Tensor,
    P: int,
) -> torch.Tensor:
    """Adjoint of graph gradient: D^T Y.

    Y: [B, E, K]  →  div: [B, P, K]
    div[j] = Σ_{e: src(e)=j} Y[e]  -  Σ_{e: dst(e)=j} Y[e]
    """
    B, E, K = Y.shape
    src, dst = edge_index[0], edge_index[1]
    div = torch.zeros((B, P, K), device=Y.device, dtype=Y.dtype)
    # + Y for source nodes, - Y for destination nodes
    div.scatter_add_(1, src.unsqueeze(0).unsqueeze(2).expand(B, E, K), Y)
    div.scatter_add_(1, dst.unsqueeze(0).unsqueeze(2).expand(B, E, K), -Y)
    return div


def _solve_semi_relaxed_ot_with_dual(
    M_c: torch.Tensor,
    alpha_c: torch.Tensor,
    b_c: torch.Tensor,
    reg: float,
    reg_mass_col: float,
    sinkhorn_iters: int,
    warmstart: "Optional[Tuple[torch.Tensor, torch.Tensor]]" = None,
    eps: float = 1e-12,
) -> "Tuple[torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]":
    """Solve semi-relaxed entropic OT and extract source dual potential.

    Semi-relaxed = exact row marginals (fi_1=1), relaxed column marginals.

    Returns
    -------
    Gamma_c : [B, P, N_c]  transport plan
    phi_c   : [B, P]  source dual potential = ε·log(u), zero-mean normalised
    warmstart_out : (logu, logv) for warm-starting next call
    """
    plan, log_dict = sinkhorn_knopp_unbalanced_torch(
        M=M_c,
        a=alpha_c,
        b=b_c,
        reg=reg,
        reg_m=(float('inf'), reg_mass_col),
        numItermax=sinkhorn_iters,
        stopThr=1e-6,
        log=True,
        warmstart=warmstart,
    )
    logu = log_dict["logu"]  # [B, P]
    logv = log_dict["logv"]  # [B, N_c]

    # Source dual potential: φ_c = ε · log(u)
    # For semi-relaxed OT with exact row marginals (fi_1=1), this is
    # exactly ∂F_c(α_c)/∂α_c.  The additive gauge constant is absorbed
    # by the simplex projection in PDHG, so we use the RAW dual.
    # NOTE: previous code zero-mean centered each class independently,
    # which destroyed cross-class level information needed for simplex
    # competition (class c₁ matching well vs c₂ matching poorly both
    # mapped to mean zero).
    phi_c = reg * logu  # [B, P]

    warmstart_out = (logu, logv)
    return plan, phi_c, warmstart_out


def _rgct_z_step_pdhg(
    G: torch.Tensor,
    Z_prev: torch.Tensor,
    edge_index: torch.Tensor,
    edge_weights: torch.Tensor,
    lambda_tv: float,
    tau_z: float,
    n_iters: int,
    eps: float = 1e-12,
) -> "Tuple[torch.Tensor, List[float]]":
    """Solve the Z-subproblem via PDHG (Chambolle-Pock).

    minimize_Z  <G, Z>  +  λ_tv · TV(Z)  +  (1/(2τ)) · ||Z - Z_prev||²
    s.t.  z_j ∈ Δ^K  for all j

    TV(Z) = Σ_e w_e ||Z[src(e)] - Z[dst(e)]||_1   (anisotropic weighted)

    Parameters
    ----------
    G         : [B, P, K]  linear cost (includes a*phi_c and clutter terms)
    Z_prev    : [B, P, K]  proximal centre
    edge_index: [2, E]  COO edge list
    edge_weights: [B, E]  per-edge weights
    lambda_tv : TV weight
    tau_z     : proximal step size
    n_iters   : number of PDHG iterations

    Returns
    -------
    Z_out     : [B, P, K]  solution
    obj_hist  : list of primal objective values per iteration
    """
    B, P, K = G.shape
    E = edge_index.size(1)
    device, dtype = G.device, G.dtype

    # Step size selection
    # ||D||² ≤ 2 * max_degree for a graph gradient operator
    if E > 0:
        # Compute max degree from edge_index
        degree = torch.zeros(P, device=device, dtype=torch.long)
        degree.scatter_add_(0, edge_index[0], torch.ones(E, device=device, dtype=torch.long))
        degree.scatter_add_(0, edge_index[1], torch.ones(E, device=device, dtype=torch.long))
        L_D = 2.0 * float(degree.max().item())
        L_D = max(L_D, 1.0)
    else:
        L_D = 1.0

    # Primal step: σ = 1 / (1/τ + λ_tv * L_D)
    # Dual step:   τ_d = 1 / (λ_tv * L_D)   (only relevant if λ_tv > 0)
    sigma_p = 1.0 / (1.0 / tau_z + lambda_tv * L_D) if lambda_tv > 0 else tau_z / (1.0 + tau_z * 0.01)
    tau_d = 1.0 / (lambda_tv * L_D) if (lambda_tv > 0 and L_D > 0) else 1.0

    # Initialise: Z = Z_prev, Y = 0
    Z = Z_prev.clone()
    Z_bar = Z.clone()
    if E > 0 and lambda_tv > 0:
        Y = torch.zeros((B, E, K), device=device, dtype=dtype)
    else:
        Y = None

    obj_hist = []

    for it in range(n_iters):
        Z_old = Z.clone()

        # --- Dual step: Y ← prox_{λ_tv * w * ||·||_1}(Y + τ_d · D · Z_bar) ---
        if Y is not None and lambda_tv > 0:
            DZ_bar = _apply_graph_gradient(Z_bar, edge_index)  # [B, E, K]
            Y = Y + tau_d * DZ_bar
            # Proximal of λ_tv * w * ||·||_1  =  soft-thresholding
            # threshold per edge: λ_tv * w_e * τ_d  — but we already folded τ_d
            # Actually for Chambolle-Pock, the dual prox is projection onto
            # the dual ball: ||Y_e / (λ_tv * w_e)||_∞ ≤ 1
            thresh = lambda_tv * edge_weights.unsqueeze(2)  # [B, E, 1]
            Y = Y.clamp(-thresh, thresh)

        # --- Primal step: Z ← proj_simplex(Z - σ(G + (Z - Z_prev)/τ + D^T Y)) ---
        grad = G + (Z - Z_prev) / tau_z
        if Y is not None and lambda_tv > 0:
            grad = grad + _apply_graph_divergence(Y, edge_index, P)
        Z = Z - sigma_p * grad
        # Project each row onto simplex
        for b_idx in range(B):
            Z[b_idx] = _project_simplex(Z[b_idx], eps=eps)

        # --- Overrelaxation ---
        Z_bar = 2.0 * Z - Z_old

        # --- Track objective ---
        obj_linear = (G * Z).sum(dim=(1, 2))  # [B]
        obj_prox = 0.5 / tau_z * ((Z - Z_prev) ** 2).sum(dim=(1, 2))  # [B]
        if E > 0 and lambda_tv > 0:
            DZ = _apply_graph_gradient(Z, edge_index)
            obj_tv = (lambda_tv * edge_weights.unsqueeze(2) * DZ.abs()).sum(dim=(1, 2))
        else:
            obj_tv = torch.zeros(B, device=device, dtype=dtype)
        obj = (obj_linear + obj_prox + obj_tv).mean().item()
        obj_hist.append(obj)

    return Z, obj_hist


@torch.no_grad()
def rgct_dual_score_batched(
    Q_batch: torch.Tensor,
    S_classes: "List[torch.Tensor]",
    b_classes: "List[torch.Tensor]",
    reg_eps: float = 0.05,
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
    eps_anneal_iters: int = 0,
    eps_anneal_start: float = 0.2,
    tv_ramp_iters: int = 0,
    verbose: bool = False,
    return_diagnostics: bool = False,
    eps: float = 1e-12,
    query_chunk_size: Optional[int] = None,
) -> "torch.Tensor | Tuple[torch.Tensor, Dict[str, Any]]":
    """RGCT-Dual: Principled reduced-objective solver.

    Prox-linear alternating minimisation over:
      J(Z) = Σ_c F_c(a ⊙ z_c) + λ_tv · TV(Z) + λ_clutter · Σ_j z_{j,0}
    s.t.  z_j ∈ Δ^{K_total}

    The Z-update uses the correct source dual potential φ_c = ∂F_c/∂α_c
    (extracted from Sinkhorn scalings) and solves the proximal subproblem
    via PDHG (Chambolle-Pock).

    Returns
    -------
    logits : [B, C]  class logits.
    """
    # Queries are independent in this solver (Sinkhorn is bmm per-b, TV graph is
    # per-query, Z is per-query). Chunking over B is a pure memory optimisation
    # with identical math; useful for large Meta-Dataset episodes (B up to ~500).
    if (
        query_chunk_size is not None
        and 0 < query_chunk_size < Q_batch.size(0)
    ):
        if return_diagnostics:
            raise NotImplementedError(
                "query_chunk_size cannot be combined with return_diagnostics=True"
            )
        logits_parts: List[torch.Tensor] = []
        for start in range(0, Q_batch.size(0), query_chunk_size):
            logits_parts.append(
                rgct_dual_score_batched(
                    Q_batch[start : start + query_chunk_size],
                    S_classes,
                    b_classes,
                    reg_eps=reg_eps,
                    reg_mass=reg_mass,
                    sinkhorn_iters=sinkhorn_iters,
                    lambda_tv=lambda_tv,
                    lambda_clutter=lambda_clutter,
                    outer_iters=outer_iters,
                    tau_z=tau_z,
                    z_pdhg_iters=z_pdhg_iters,
                    use_clutter=use_clutter,
                    anisotropic_tv=anisotropic_tv,
                    rgct_scoring=rgct_scoring,
                    eps_anneal_iters=eps_anneal_iters,
                    eps_anneal_start=eps_anneal_start,
                    tv_ramp_iters=tv_ramp_iters,
                    verbose=verbose,
                    return_diagnostics=False,
                    eps=eps,
                    query_chunk_size=None,
                )
            )
        return torch.cat(logits_parts, dim=0)

    B, P, D = Q_batch.shape
    C = len(S_classes)
    device = Q_batch.device
    dtype = torch.float64
    # Store [B, P, N_c] cost matrices and transport plans in float32 to halve
    # their memory footprint; upcast to float64 only at the Sinkhorn boundary
    # where exp(-M/reg_eps) needs float64 to avoid underflow at small reg_eps.
    store_dtype = torch.float32

    K_total = C + 1 if use_clutter else C
    clutter_offset = 1 if use_clutter else 0

    # Query marginal
    a = torch.full((B, P), 1.0 / float(P), device=device, dtype=dtype)

    # Build cost matrices and normalised support marginals
    M_list: "List[torch.Tensor]" = []
    b_list: "List[torch.Tensor]" = []
    for c in range(C):
        S_c = S_classes[c]
        N_c = S_c.size(0)
        sim = torch.clamp(Q_batch.float() @ S_c.float().t(), -1.0, 1.0)
        M_c = (1.0 - sim).to(store_dtype)  # [B, P, N_c]
        b_c = b_classes[c].to(device=device, dtype=dtype).unsqueeze(0).expand(B, N_c)
        total_b = b_c.sum(dim=1, keepdim=True).clamp(min=eps)
        b_c = b_c * (a.sum(dim=1, keepdim=True) / total_b)
        M_list.append(M_c)
        b_list.append(b_c)

    # Build grid edges for TV
    edge_index = _build_grid_edges(P, device)

    # Compute anisotropic edge weights
    if anisotropic_tv and edge_index.size(1) > 0:
        Q64 = Q_batch.to(dtype)
        all_edge_weights = []
        for b_idx in range(B):
            src_feats = Q64[b_idx, edge_index[0]]
            dst_feats = Q64[b_idx, edge_index[1]]
            dists = ((src_feats - dst_feats) ** 2).sum(dim=1)
            median_d = dists.median().clamp(min=eps)
            w = torch.exp(-dists / (2.0 * median_d))
            all_edge_weights.append(w)
        batch_edge_weights = torch.stack(all_edge_weights)  # [B, E]
    else:
        batch_edge_weights = torch.ones((B, edge_index.size(1)),
                                         device=device, dtype=dtype)

    # Initialise Z: uniform allocation
    Z = torch.full((B, P, K_total), 1.0 / float(K_total),
                    device=device, dtype=dtype)

    # Warm-start storage for OT solves
    ot_warmstarts: "List[Optional[Tuple[torch.Tensor, torch.Tensor]]]" = [None] * C

    # Diagnostics storage
    diag_Z_history: List[torch.Tensor] = []
    diag_obj_history: List[Dict[str, float]] = []

    # ── Alternating optimisation ──
    plans: "List[Optional[torch.Tensor]]" = [None] * C
    phi_all: "List[Optional[torch.Tensor]]" = [None] * C

    for outer in range(int(outer_iters)):

        # ── Continuation / annealing ──
        if eps_anneal_iters > 0 and outer < eps_anneal_iters:
            t = float(outer) / float(eps_anneal_iters)
            reg_eff = eps_anneal_start * (1.0 - t) + reg_eps * t
        else:
            reg_eff = reg_eps

        if tv_ramp_iters > 0 and outer < tv_ramp_iters:
            t = float(outer) / float(tv_ramp_iters)
            tv_eff = lambda_tv * t
        else:
            tv_eff = lambda_tv

        # ── BLOCK 1: OT solve + dual extraction (Z fixed) ──
        total_ot_value = torch.zeros(B, device=device, dtype=dtype)
        for c in range(C):
            z_c = Z[:, :, c + clutter_offset].clamp(min=1e-8)  # [B, P]
            alpha_c = a * z_c  # [B, P]   row marginal for class c

            Gamma_c, phi_c, ws_out = _solve_semi_relaxed_ot_with_dual(
                M_c=M_list[c].to(dtype),
                alpha_c=alpha_c,
                b_c=b_list[c],
                reg=reg_eff,
                reg_mass_col=reg_mass,
                sinkhorn_iters=sinkhorn_iters,
                warmstart=ot_warmstarts[c],
                eps=eps,
            )
            Gamma_c = Gamma_c.to(store_dtype)
            plans[c] = Gamma_c
            phi_all[c] = phi_c
            ot_warmstarts[c] = ws_out

            # F_c value = <M_c, Gamma_c> + ε · KL(Gamma_c | c_mat)  (approx by primal cost)
            # Compute the dot product in store_dtype to avoid materialising
            # a float64 [B, P, N_c] temporary.
            total_ot_value = total_ot_value + (Gamma_c * M_list[c]).sum(dim=(1, 2)).to(dtype)

        # ── BLOCK 2: Z update via PDHG ──
        # Build gradient G[:,:,c+offset] = a * phi_c
        G = torch.zeros((B, P, K_total), device=device, dtype=dtype)
        for c in range(C):
            G[:, :, c + clutter_offset] = a * phi_all[c]

        if use_clutter:
            G[:, :, 0] = G[:, :, 0] + lambda_clutter

        Z_prev = Z.clone()
        Z, pdhg_obj_hist = _rgct_z_step_pdhg(
            G=G,
            Z_prev=Z_prev,
            edge_index=edge_index,
            edge_weights=batch_edge_weights,
            lambda_tv=tv_eff,
            tau_z=tau_z,
            n_iters=z_pdhg_iters,
            eps=eps,
        )

        # ── Logging / diagnostics ──
        # Compute diagnostic values (always, for diagnostics; print only if verbose)
        if edge_index.size(1) > 0 and tv_eff > 0:
            DZ = _apply_graph_gradient(Z, edge_index)
            tv_val = (tv_eff * batch_edge_weights.unsqueeze(2) * DZ.abs()).sum(dim=(1, 2)).mean().item()
        else:
            tv_val = 0.0
        clutter_mass = Z[:, :, 0].sum(dim=1).mean().item() if use_clutter else 0.0
        z_change = ((Z - Z_prev) ** 2).sum(dim=(1, 2)).sqrt().mean().item()

        if return_diagnostics:
            diag_Z_history.append(Z.detach().cpu().clone())
            diag_obj_history.append({
                "outer": outer,
                "ot_value": total_ot_value.mean().item(),
                "tv_value": tv_val,
                "clutter_mass": clutter_mass,
                "z_change": z_change,
                "pdhg_final_obj": pdhg_obj_hist[-1] if pdhg_obj_hist else 0.0,
            })

        if verbose:
            fc_vals = []
            for c in range(C):
                fc = (plans[c] * M_list[c]).sum(dim=(1, 2)).mean().item()
                fc_vals.append(fc)

            print(f"  RGCT-Dual outer={outer}: "
                  f"OT={total_ot_value.mean().item():.4f} "
                  f"TV={tv_val:.4f} "
                  f"clutter={clutter_mass:.4f} "
                  f"||ΔZ||={z_change:.6f} "
                  f"F_c={[f'{v:.4f}' for v in fc_vals]} "
                  f"PDHG_obj={pdhg_obj_hist[-1]:.4f}" if pdhg_obj_hist else "")

    # ── Final transport solve with converged Z ──
    for c in range(C):
        z_c = Z[:, :, c + clutter_offset].clamp(min=1e-8)
        alpha_c = a * z_c
        final_plan, _, _ = _solve_semi_relaxed_ot_with_dual(
            M_c=M_list[c].to(dtype),
            alpha_c=alpha_c,
            b_c=b_list[c],
            reg=reg_eps,
            reg_mass_col=reg_mass,
            sinkhorn_iters=sinkhorn_iters,
            warmstart=ot_warmstarts[c],
            eps=eps,
        )
        plans[c] = final_plan.to(store_dtype)

    # ── Compute logits ──
    scoring = str(rgct_scoring).strip().lower()
    primal_costs = torch.zeros((B, C), device=device, dtype=dtype)
    class_masses = torch.zeros((B, C), device=device, dtype=dtype)

    for c in range(C):
        primal_costs[:, c] = (plans[c] * M_list[c]).sum(dim=(1, 2)).to(dtype)
        class_masses[:, c] = plans[c].sum(dim=(1, 2)).to(dtype)

    if scoring == "primal":
        logits = -primal_costs
    elif scoring == "mass":
        logits = torch.log(class_masses.clamp(min=eps))
    elif scoring == "hybrid":
        logits = (torch.log(class_masses.clamp(min=eps))
                  - 0.5 * primal_costs / primal_costs.abs().mean(dim=1, keepdim=True).clamp(min=eps))
    else:
        logits = -primal_costs

    logits_out = logits.to(torch.float32)

    if return_diagnostics:
        diagnostics = {
            "Z_final": Z.detach().cpu().clone(),
            "Z_history": diag_Z_history,
            "obj_history": diag_obj_history,
            "primal_costs": primal_costs.detach().cpu().clone(),
            "class_masses": class_masses.detach().cpu().clone(),
        }
        return logits_out, diagnostics

    return logits_out


def _sinkhorn_balanced_torch(
    M: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    reg: float,
    numItermax: int = 500,
    stopThr: float = 1e-6,
) -> torch.Tensor:
    """Balanced Sinkhorn for 2D cost matrix. Thin wrapper."""
    from utils.sinkhorn_unbalanced_torch import sinkhorn_balanced_torch
    return sinkhorn_balanced_torch(M, a, b, reg=reg, numItermax=numItermax, stopThr=stopThr)


def _normalize_branch_logits(logits: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    centered = logits - logits.mean(dim=1, keepdim=True)
    scale = centered.std(dim=1, keepdim=True, unbiased=False).clamp_min(eps)
    return centered / scale


def _top2_margin(logits: torch.Tensor) -> torch.Tensor:
    if logits.size(1) <= 1:
        return logits.squeeze(1)
    top2 = torch.topk(logits, k=2, dim=1).values
    return top2[:, 0] - top2[:, 1]


def _blend_global_parts_logits(
    global_logits: torch.Tensor,
    parts_logits: torch.Tensor,
    alpha_global: float,
    blend_mode: str = "fixed",
    blend_margin_temp: float = 0.2,
    override_global_margin_thresh: float = 0.1,
    override_parts_delta_thresh: float = 0.0,
    eps: float = 1e-12,
) -> torch.Tensor:
    alpha = float(np.clip(alpha_global, 0.0, 1.0))
    mode = str(blend_mode).strip().lower()
    if mode == "fixed":
        return (alpha * global_logits + (1.0 - alpha) * parts_logits).to(torch.float32)
    if mode == "top2_override":
        if global_logits.size(1) <= 1:
            return global_logits.to(torch.float32)
        top2_vals, top2_idx = torch.topk(global_logits.to(torch.float32), k=2, dim=1)
        global_margin = top2_vals[:, 0] - top2_vals[:, 1]
        batch_idx = torch.arange(global_logits.size(0), device=global_logits.device)
        runner_delta = (
            parts_logits[batch_idx, top2_idx[:, 1]] - parts_logits[batch_idx, top2_idx[:, 0]]
        ).to(torch.float32)
        override = (
            (global_margin <= float(override_global_margin_thresh))
            & (runner_delta >= float(override_parts_delta_thresh))
        )
        blended = global_logits.to(torch.float32).clone()
        if override.any():
            override_idx = torch.nonzero(override, as_tuple=False).squeeze(1)
            winners = top2_idx[override, 0]
            runners = top2_idx[override, 1]
            winner_vals = top2_vals[override, 0]
            runner_vals = top2_vals[override, 1]
            boost = runner_delta[override].clamp_min(float(eps))
            blended[override_idx, winners] = runner_vals
            blended[override_idx, runners] = winner_vals + boost
        return blended
    if mode != "adaptive_margin":
        raise ValueError(f"Unknown blend_mode '{blend_mode}'")

    if alpha <= eps:
        return parts_logits.to(torch.float32)
    if alpha >= 1.0 - eps:
        return global_logits.to(torch.float32)

    global_norm = _normalize_branch_logits(global_logits.to(torch.float32), eps=eps)
    parts_norm = _normalize_branch_logits(parts_logits.to(torch.float32), eps=eps)

    global_margin = _top2_margin(global_norm)
    parts_margin = _top2_margin(parts_norm)
    temp = max(float(blend_margin_temp), eps)

    prior_global = float(np.clip(alpha, eps, 1.0 - eps))
    prior_local = 1.0 - prior_global
    prior_logit = math.log(prior_local / prior_global)
    gate_local = torch.sigmoid(
        torch.tensor(prior_logit, device=global_norm.device, dtype=global_norm.dtype)
        + (parts_margin - global_margin) / temp
    ).unsqueeze(1)
    return ((1.0 - gate_local) * global_norm + gate_local * parts_norm).to(torch.float32)


def _calibrate_episode_logits(
    logits: torch.Tensor,
    n_way: int,
    reg_eps: float,
    eps: float = 1e-12,
) -> torch.Tensor:
    bq = logits.size(0)
    Mqc = -logits.to(torch.float64)
    a_q = torch.full((bq,), 1.0 / float(bq), device=logits.device, dtype=torch.float64)
    b_c = torch.full((n_way,), 1.0 / float(n_way), device=logits.device, dtype=torch.float64)
    gamma_qc = _sinkhorn_balanced_torch(
        Mqc,
        a_q,
        b_c,
        reg=float(reg_eps),
        numItermax=500,
        stopThr=1e-6,
    ).to(torch.float32)
    return logits + torch.log(gamma_qc + eps)


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
        blend_mode: str = "fixed",
        blend_margin_temp: float = 0.2,
        override_global_margin_thresh: float = 0.1,
        override_parts_delta_thresh: float = 0.0,
        calibrate_episode: bool = True,
        episodic_trans_mode: str = "support",
        episodic_trans_eps: float = 1e-6,
        lambda_tv: float = 0.1,
        lambda_clutter: float = 0.5,
        rgct_outer_iters: int = 5,
        tau_z: float = 0.1,
        z_pdhg_iters: int = 50,
        use_clutter: bool = True,
        anisotropic_tv: bool = True,
        rgct_scoring: str = "primal",
        eps_anneal_iters: int = 0,
        eps_anneal_start: float = 0.2,
        tv_ramp_iters: int = 0,
        verbose_rgct: bool = False,
        scoring_mode: str = "rgct_dual",
        max_patches: int = 0,
        query_chunk_size: Optional[int] = None,
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
        self.blend_mode = str(blend_mode).strip().lower()
        self.blend_margin_temp = float(blend_margin_temp)
        self.override_global_margin_thresh = float(override_global_margin_thresh)
        self.override_parts_delta_thresh = float(override_parts_delta_thresh)
        self.calibrate_episode = bool(calibrate_episode)
        self.episodic_trans_mode = str(episodic_trans_mode).strip().lower()
        self.episodic_trans_eps = float(episodic_trans_eps)
        self.lambda_tv = float(lambda_tv)
        self.lambda_clutter = float(lambda_clutter)
        self.rgct_outer_iters = int(rgct_outer_iters)
        self.tau_z = float(tau_z)
        self.z_pdhg_iters = int(z_pdhg_iters)
        self.use_clutter = bool(use_clutter)
        self.anisotropic_tv = bool(anisotropic_tv)
        self.rgct_scoring = str(rgct_scoring).strip().lower()
        self.eps_anneal_iters = int(eps_anneal_iters)
        self.eps_anneal_start = float(eps_anneal_start)
        self.tv_ramp_iters = int(tv_ramp_iters)
        self.verbose_rgct = bool(verbose_rgct)
        self.max_patches = int(max_patches)
        self.query_chunk_size = (
            int(query_chunk_size) if query_chunk_size is not None and int(query_chunk_size) > 0 else None
        )
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
        std = images.std(dim=(0, 2, 3), keepdim=True, unbiased=False).clamp_min(self.episodic_trans_eps)
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
            query_tokens,
            self.support_classes,
            self.support_masses,
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
            eps_anneal_iters=self.eps_anneal_iters,
            eps_anneal_start=self.eps_anneal_start,
            tv_ramp_iters=self.tv_ramp_iters,
            verbose=self.verbose_rgct,
            eps=self.eps,
            query_chunk_size=self.query_chunk_size,
        )

        centers = []
        for class_id in torch.unique(self.support_labels_cached).tolist():
            idx = torch.nonzero(self.support_labels_cached == class_id, as_tuple=False).squeeze(1)
            centers.append(F.normalize(support_whole[idx].mean(dim=0, keepdim=True), dim=1).squeeze(0))
        self.global_prototypes = torch.stack(centers, dim=0).to(device)

        global_logits = query_whole @ self.global_prototypes.t()
        if self.blend_mode == "top2_override" and self.calibrate_episode:
            global_logits = _calibrate_episode_logits(logits=global_logits, n_way=self.n_way, reg_eps=self.reg_eps, eps=self.eps)
        logits = _blend_global_parts_logits(
            global_logits=global_logits, parts_logits=parts_logits,
            alpha_global=self.alpha_global, blend_mode=self.blend_mode,
            blend_margin_temp=self.blend_margin_temp,
            override_global_margin_thresh=self.override_global_margin_thresh,
            override_parts_delta_thresh=self.override_parts_delta_thresh,
            eps=self.eps,
        )
        if self.calibrate_episode and self.blend_mode != "top2_override":
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
