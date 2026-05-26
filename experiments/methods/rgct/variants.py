"""Single RGCT-Dual variant registry."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Type

from easyfsl.methods import FewShotClassifier

from .method import RGCTDualNet


@dataclass(frozen=True)
class VariantSpec:
    name: str
    method_class: Type[FewShotClassifier]
    params: Dict[str, Any]
    motivation: str


RGCT_DUAL_V9_SHARP_PARAMS: Dict[str, Any] = {
    "reg_eps": 0.02,
    "reg_mass": 0.3,
    "sinkhorn_iters": 200,
    "use_ctb": True,
    "n_support_atoms": 64,
    "bary_iters": 5,
    "bary_inner_max": 30,
    "support_mix": 0.5,
    "support_trim_ratio": 0.95,
    "support_gate_temp": 0.5,
    "support_num_iter": 120,
    "alpha_global": 0.4,
    "calibrate_episode": True,
    "episodic_trans_mode": "support",
    "scoring_mode": "rgct_dual",
    "lambda_tv": 0.1,
    "lambda_clutter": 0.5,
    "rgct_outer_iters": 5,
    "tau_z": 0.1,
    "z_pdhg_iters": 50,
    "use_clutter": True,
    "anisotropic_tv": True,
    "rgct_scoring": "primal",
}


VARIANTS: List[VariantSpec] = [
    VariantSpec(
        name="rgct_dual_v9_sharp",
        method_class=RGCTDualNet,
        params=RGCT_DUAL_V9_SHARP_PARAMS,
        motivation=(
            "RGCT-Dual with sharper OT (reg_eps=0.02), CTB support, clutter, "
            "anisotropic TV, and primal scoring."
        ),
    )
]


def get_variant_map() -> Dict[str, VariantSpec]:
    return {variant.name: variant for variant in VARIANTS}


__all__ = [
    "RGCT_DUAL_V9_SHARP_PARAMS",
    "VARIANTS",
    "VariantSpec",
    "get_variant_map",
]
