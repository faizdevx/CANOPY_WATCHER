#!/usr/bin/env python3
"""
edge/config/loader.py — Canopy Watch config loader

Resolves the final runtime configuration by layering, in this order (later wins):

    base.yaml -> <profile>.yaml -> hardware.generated.yaml -> env vars

- base.yaml                  common defaults, always present
- <profile>.yaml             project's behavioral decisions (laptop/rpi/jetson_nano/jetson_orin)
- hardware.generated.yaml    machine-detected facts (optional, written by scripts/detect_hardware.py)
- env vars                   secrets (via *_env indirection) + ad-hoc overrides

Business logic should never read YAML files directly — always go through get_config().

Library usage:
    from edge.config.loader import get_config
    cfg = get_config()
    cfg["hardware"]["driver"]["camera"]

CLI usage (debugging):
    python -m edge.config.loader --profile laptop
    CW_PROFILE=jetson_orin python -m edge.config.loader

Env var conventions:
    CW_PROFILE           selects <profile>.yaml (laptop | rpi | jetson_nano | jetson_orin)
    CW_CONFIG_DIR         overrides the config directory (default: ./config)
    CW_NODE_ID / CW_ORG_ID / CW_SITE / CW_ZONE
                          convenience overrides for identity.*
    CW__section__key=val  generic override for any nested path, e.g.
                          CW__networking__sync__endpoint=mqtt://gateway.local:1883
    *_env keys in yaml    e.g. `auth_token_env: CW_SYNC_TOKEN` — loader reads
                          CW_SYNC_TOKEN and exposes it as `auth_token`. Actual
                          secret values are never written to any yaml file.
"""

import argparse
import copy
import os
import sys
from pathlib import Path

import yaml


class ConfigError(Exception):
    """Raised when config is missing, malformed, or fails validation."""


DEFAULT_CONFIG_DIR = "config"
ENV_OVERRIDE_PREFIX = "CW__"
ENV_OVERRIDE_DELIM = "__"

SIMPLE_ENV_MAP = {
    "CW_NODE_ID": "identity.node_id",
    "CW_ORG_ID": "identity.org_id",
    "CW_SITE": "identity.site",
    "CW_ZONE": "identity.zone",
}

# Paths that must resolve to a non-null, non-empty value once all layers
# (including env vars) are applied. Keeps a misconfigured node from booting.
REQUIRED_PATHS = [
    "identity.node_id",
    "hardware.hardware_tier",
    "hardware.driver.camera",
    "ai.accelerator",
    "ai.tier1.model_path",
    "ai.tier2.mode",
    "storage.event_store.path",
    "networking.event_bus.type",
    "networking.sync.protocol",
]

ENUM_CONSTRAINTS = {
    "hardware.hardware_tier": {"constrained", "accelerated"},
    "ai.accelerator": {"cpu", "cuda", "tensorrt", "edgetpu"},
    "ai.tier2.mode": {"local", "defer_to_cloud", "disabled"},
    "networking.event_bus.type": {"inprocess", "redis", "mqtt", "kafka"},
    "networking.sync.protocol": {"none", "http", "mqtt", "kafka"},
}


# ---------- yaml loading + merge ----------

def load_yaml(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path) as f:
        data = yaml.safe_load(f)
    return data or {}


def deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base. Override wins on scalars/lists.
    Dicts merge key-by-key; anything else (lists included) is replaced
    wholesale, never concatenated — keeps merge behavior predictable."""
    result = copy.deepcopy(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


# ---------- env var overrides ----------

def _set_by_path(d: dict, dotted_path: str, value) -> None:
    parts = dotted_path.split(".")
    cur = d
    for part in parts[:-1]:
        cur = cur.setdefault(part, {})
    cur[parts[-1]] = value


def _coerce(value: str):
    """Best-effort type coercion for env var override values (so
    CW__ai__tier2__enabled=true becomes a real bool, not the string 'true')."""
    try:
        return yaml.safe_load(value)
    except yaml.YAMLError:
        return value


def apply_env_overrides(cfg: dict) -> dict:
    """Applies, in order: (1) SIMPLE_ENV_MAP convenience vars, then
    (2) generic CW__section__key=value overrides. Generic overrides win if
    both target the same path. This is the last layer before validation."""
    cfg = copy.deepcopy(cfg)

    for env_var, dotted_path in SIMPLE_ENV_MAP.items():
        if env_var in os.environ:
            _set_by_path(cfg, dotted_path, os.environ[env_var])

    for env_key, env_value in os.environ.items():
        if not env_key.startswith(ENV_OVERRIDE_PREFIX):
            continue
        dotted_path = ".".join(env_key[len(ENV_OVERRIDE_PREFIX):].split(ENV_OVERRIDE_DELIM))
        _set_by_path(cfg, dotted_path, _coerce(env_value))

    return cfg


# ---------- secret resolution ----------

def resolve_secrets(cfg: dict, warnings: list) -> dict:
    """Walks the config; any key ending in `_env` names an env var holding the
    real value. Replaces it with a sibling key (suffix stripped) set to that
    value, or None + a warning if the env var isn't set. Config files never
    contain the secret itself, only the name of the var that holds it."""

    def walk(node):
        if not isinstance(node, dict):
            return node
        resolved = {}
        for key, value in node.items():
            if key.endswith("_env"):
                real_key = key[: -len("_env")]
                if isinstance(value, str):
                    secret_value = os.environ.get(value)
                    if secret_value is None:
                        warnings.append(
                            f"env var '{value}' referenced by '{key}' is not set — "
                            f"'{real_key}' will be null"
                        )
                else:
                    secret_value = None  # no env var name given (e.g. laptop profile)
                resolved[real_key] = secret_value
            else:
                resolved[key] = walk(value) if isinstance(value, dict) else value
        return resolved

    return walk(cfg)


# ---------- validation ----------

def _get_by_path(d: dict, dotted_path: str):
    cur = d
    for part in dotted_path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None, False
        cur = cur[part]
    return cur, True


def validate(cfg: dict) -> list:
    """Returns a list of human-readable error strings; empty list = valid.
    Fails fast and loud rather than silently falling back to defaults that
    could mask a misconfigured field node."""
    errors = []

    for path in REQUIRED_PATHS:
        value, present = _get_by_path(cfg, path)
        if not present or value in (None, ""):
            errors.append(f"missing required config: {path}")

    for path, allowed in ENUM_CONSTRAINTS.items():
        value, present = _get_by_path(cfg, path)
        if present and value is not None and value not in allowed:
            errors.append(f"invalid value for {path}: '{value}' (expected one of {sorted(allowed)})")

    return errors


# ---------- public entrypoint ----------

_CACHED_CONFIG = None


def load_config(profile: str = None, config_dir: str = None, use_cache: bool = True) -> dict:
    global _CACHED_CONFIG
    if use_cache and _CACHED_CONFIG is not None:
        return _CACHED_CONFIG

    config_dir_path = Path(config_dir or os.environ.get("CW_CONFIG_DIR", DEFAULT_CONFIG_DIR))
    profile = profile or os.environ.get("CW_PROFILE")

    if not profile:
        raise ConfigError(
            "no profile specified — set CW_PROFILE env var or pass profile= "
            "(e.g. laptop, rpi, jetson_nano, jetson_orin)"
        )

    base_path = config_dir_path / "base.yaml"
    profile_path = config_dir_path / f"{profile}.yaml"
    hardware_path = config_dir_path / "hardware.generated.yaml"

    if not base_path.exists():
        raise ConfigError(f"base config not found: {base_path}")
    if not profile_path.exists():
        raise ConfigError(f"profile config not found: {profile_path}")
    if not hardware_path.exists():
        print(
            f"warning: {hardware_path} not found — run scripts/detect_hardware.py to "
            f"auto-detect hardware facts. Continuing with profile defaults only.",
            file=sys.stderr,
        )

    merged = load_yaml(base_path)
    merged = deep_merge(merged, load_yaml(profile_path))
    merged = deep_merge(merged, load_yaml(hardware_path))
    merged = apply_env_overrides(merged)

    warnings = []
    merged = resolve_secrets(merged, warnings)
    for w in warnings:
        print(f"warning: {w}", file=sys.stderr)

    errors = validate(merged)
    if errors:
        raise ConfigError("config validation failed:\n  - " + "\n  - ".join(errors))

    if use_cache:
        _CACHED_CONFIG = merged
    return merged


def get_config() -> dict:
    """Convenience accessor for application code. Reads CW_PROFILE / CW_CONFIG_DIR
    from the environment. Result is cached after the first successful load."""
    return load_config()


# ---------- CLI ----------

def _mask_secrets_for_print(cfg: dict) -> dict:
    """Never dump raw secret values to stdout/logs — mask anything under a
    key that looks sensitive (heuristic: token/secret/password/key)."""
    sensitive_markers = ("token", "secret", "password", "key")

    def mask(key, value):
        if value and any(m in key.lower() for m in sensitive_markers):
            s = str(value)
            return s[:2] + "***" if len(s) > 2 else "***"
        return value

    def walk(node):
        if isinstance(node, dict):
            return {k: (walk(v) if isinstance(v, dict) else mask(k, v)) for k, v in node.items()}
        return node

    return walk(cfg)


def main():
    parser = argparse.ArgumentParser(description="Load and validate Canopy Watch config")
    parser.add_argument("--profile", default=None, help="laptop | rpi | jetson_nano | jetson_orin")
    parser.add_argument("--config-dir", default=None)
    args = parser.parse_args()

    try:
        cfg = load_config(profile=args.profile, config_dir=args.config_dir, use_cache=False)
    except ConfigError as e:
        print(f"ConfigError: {e}", file=sys.stderr)
        sys.exit(1)

    print(yaml.safe_dump(_mask_secrets_for_print(cfg), sort_keys=False))


if __name__ == "__main__":
    main()