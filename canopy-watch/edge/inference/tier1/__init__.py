"""Tier1 backend registry and factory."""

from typing import Dict, Tuple, Type

from .base import Tier1Detector, Tier1Error
from .heuristic import HeuristicTier1
from .tflite_ssd import TFLiteSSDTier1


TIER1_REGISTRY: Dict[str, Type[Tier1Detector]] = {
    "heuristic": HeuristicTier1,
    "tflite_ssd": TFLiteSSDTier1,
}


def available_tier1_backends() -> Tuple[str, ...]:
    """Return Tier1 backend names in deterministic order."""
    return tuple(TIER1_REGISTRY.keys())


def load_tier1_class(backend: str) -> Type[Tier1Detector]:
    """Return the Tier1 class registered under ``backend``."""
    try:
        return TIER1_REGISTRY[backend]
    except KeyError as exc:
        available = ", ".join(available_tier1_backends())
        raise Tier1Error(
            f"unknown tier1 backend {backend!r}; "
            f"available: {available}"
        ) from exc


def create_tier1(
    backend="heuristic",
    params=None,
) -> Tier1Detector:
    if isinstance(backend, dict):
        cfg = backend
        tier1_cfg = cfg.get("ai", {}).get("tier1", {})
        backend = tier1_cfg.get("backend", "heuristic")
        params = dict(tier1_cfg)
        params.pop("backend", None)

    cls = load_tier1_class(backend)
    return cls(params)

__all__ = [
    "TIER1_REGISTRY",
    "Tier1Detector",
    "Tier1Error",
    "HeuristicTier1",
    "TFLiteSSDTier1",
    "available_tier1_backends",
    "load_tier1_class",
    "create_tier1",
]