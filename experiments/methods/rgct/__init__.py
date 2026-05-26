"""Minimal RGCT-Dual package."""

from .method import RGCTDualNet, rgct_dual_score_batched
from .variants import RGCT_DUAL_V9_SHARP_PARAMS, VARIANTS, get_variant_map

__all__ = [
    "RGCTDualNet",
    "RGCT_DUAL_V9_SHARP_PARAMS",
    "VARIANTS",
    "get_variant_map",
    "rgct_dual_score_batched",
]
