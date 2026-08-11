"""Tier2 backend registry, factory, and mode helpers."""

from typing import Dict, Tuple, Type

from .base import (
    HeuristicTier2,
    OnnxTier2,
    Tier2Error,
    Tier2Verifier,
)


TIER2_REGISTRY: Dict[str, Type[Tier2Verifier]] = {
    "heuristic": HeuristicTier2,
    "onnx": OnnxTier2,
}


def available_tier2_backends() -> Tuple[str, ...]:
    """Return Tier2 backend names in deterministic order."""
    return tuple(TIER2_REGISTRY.keys())


def load_tier2_class(backend: str) -> Type[Tier2Verifier]:
    """Return the Tier2 class registered under ``backend``."""
    try:
        return TIER2_REGISTRY[backend]
    except KeyError as exc:
        available = ", ".join(available_tier2_backends())
        raise Tier2Error(
            f"unknown tier2 backend {backend!r}; "
            f"available: {available}"
        ) from exc


def create_tier2(
    backend="heuristic",
    params=None,
) -> Tier2Verifier:
    if isinstance(backend, dict):
        cfg = backend
        tier2_cfg = cfg.get("ai", {}).get("tier2", {})
        backend = tier2_cfg.get("backend", "heuristic")
        params = dict(tier2_cfg)
        params.pop("backend", None)

    cls = load_tier2_class(backend)
    return cls(params)


def tier2_mode(config) -> str:
    """Return the configured Tier2 execution mode."""
    ai = config.get("ai") or {}
    tier2 = ai.get("tier2") or {}
    mode = tier2.get("mode", "local")

    valid = {"local", "defer_to_cloud", "disabled"}
    if mode not in valid:
        raise Tier2Error(
            f"invalid tier2 mode {mode!r}; "
            f"expected one of: {', '.join(sorted(valid))}"
        )

    return mode


__all__ = [
    "TIER2_REGISTRY",
    "Tier2Verifier",
    "Tier2Error",
    "HeuristicTier2",
    "OnnxTier2",
    "available_tier2_backends",
    "load_tier2_class",
    "create_tier2",
    "tier2_mode",
]