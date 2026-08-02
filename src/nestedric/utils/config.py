"""YAML config loading with inheritance (``_base_``) and CLI overrides.

Every experiment is a config; nothing about a run may be hard-coded, because a number
in the paper has to be traceable to a file under version control. The rules are:

* ``_base_`` names a sibling config to inherit from, resolved relative to the child.
  Inheritance is a deep merge: the child wins key by key, so ``method/ewc.yaml`` can
  state only ``lambda_ewc`` and pick up the optimiser from ``finetune.yaml``.
* Overrides are ``dotted.key=value`` strings, parsed as YAML scalars so ``true``,
  ``3``, ``1e-3`` and ``[1, 32]`` all arrive as the right type.
* An override of a key that does not exist raises. A typo in ``--set`` should not
  silently do nothing for six hours of GPU time.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


class ConfigError(ValueError):
    """Raised for malformed configs, unresolvable ``_base_``, or unknown override keys."""


def _deep_merge(base: dict, child: dict) -> dict:
    """Recursively merge *child* into *base*; child scalars and lists win outright."""
    out = deepcopy(base)
    for key, value in child.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = deepcopy(value)
    return out


def _read_yaml(path: Path) -> dict:
    if not path.exists():
        raise ConfigError(f"config not found: {path}")
    with path.open() as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ConfigError(f"{path} must contain a mapping, got {type(data).__name__}")
    return data


def _resolve_bases(path: Path, seen: tuple[Path, ...] = ()) -> dict:
    """Load *path*, recursively merging its ``_base_`` ancestors first."""
    path = path.resolve()
    if path in seen:
        chain = " -> ".join(p.name for p in (*seen, path))
        raise ConfigError(f"circular _base_ inheritance: {chain}")

    data = _read_yaml(path)
    base_ref = data.pop("_base_", None)
    if base_ref is None:
        return data

    base_path = (path.parent / str(base_ref)).resolve()
    base = _resolve_bases(base_path, (*seen, path))
    return _deep_merge(base, data)


def _parse_scalar(text: str) -> Any:
    """Parse an override value as YAML, falling back to the raw string."""
    try:
        return yaml.safe_load(text)
    except yaml.YAMLError:
        return text


def _apply_override(cfg: dict, dotted: str, value: Any) -> None:
    """Set ``cfg[a][b][c] = value`` for ``dotted='a.b.c'``, requiring the key to exist."""
    parts = dotted.split(".")
    node: Any = cfg
    for part in parts[:-1]:
        if not isinstance(node, dict) or part not in node:
            raise ConfigError(f"override {dotted!r}: no such key {part!r}")
        node = node[part]
    leaf = parts[-1]
    if not isinstance(node, dict) or leaf not in node:
        raise ConfigError(
            f"override {dotted!r}: no such key {leaf!r} "
            f"(available: {sorted(node) if isinstance(node, dict) else type(node).__name__})"
        )
    node[leaf] = value


def load_config(path: str | Path, overrides: list[str] | None = None) -> dict:
    """Load a YAML config, resolve ``_base_`` inheritance, apply ``key=value`` overrides.

    Parameters
    ----------
    path
        Path to the YAML config.
    overrides
        ``dotted.key=value`` strings, applied in order after inheritance resolves.

    Returns
    -------
    dict
        The merged config, with ``_config_path`` recording its origin so run
        directories can say where their settings came from.

    """
    cfg = _resolve_bases(Path(path))

    for item in overrides or []:
        if "=" not in item:
            raise ConfigError(f"override {item!r} is not of the form key=value")
        dotted, _, raw = item.partition("=")
        _apply_override(cfg, dotted.strip(), _parse_scalar(raw.strip()))

    cfg["_config_path"] = str(Path(path))
    return cfg


def apply_config_overrides(cfg: dict, overrides: dict[str, Any]) -> dict:
    """Apply an ``{"dotted.key": value}`` mapping, as used by ``experiment.overrides``.

    Returns a copy; the caller's config is left alone so a sweep can reuse it.
    """
    out = deepcopy(cfg)
    for dotted, value in overrides.items():
        _apply_override(out, dotted, value)
    return out
