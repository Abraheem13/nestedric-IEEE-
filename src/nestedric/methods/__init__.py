"""Continual-learning methods. All expose the same `Method` protocol."""

from __future__ import annotations

import importlib

REGISTRY: dict[str, type] = {}

#: Modules holding registered methods. Imported on first lookup so that registration
#: does not depend on import order elsewhere -- a method missing from a sweep because
#: nobody imported its module is a silently smaller experiment.
_MODULES = (
    "finetune",
    "joint",
    "ewc",
    "si",
    "replay",
    "agem",
    "lwf",
    "bilevel",
    "titans",
    "nestedric",
)


def register(name: str):
    """Class decorator adding a method to the registry used by the CLI."""

    def _wrap(cls):
        REGISTRY[name] = cls
        return cls

    return _wrap


def load_all() -> dict[str, type]:
    """Import every method module, ignoring those still stubbed out."""
    for mod in _MODULES:
        try:
            importlib.import_module(f"nestedric.methods.{mod}")
        except NotImplementedError:  # pragma: no cover - a stub not yet written
            continue
    return REGISTRY


def build_method(name: str, model, cfg: dict, device: str = "cpu"):
    """Instantiate a registered method by name."""
    load_all()
    if name not in REGISTRY:
        raise KeyError(f"unknown method {name!r}; registered: {sorted(REGISTRY)}")
    return REGISTRY[name](model, cfg, device)
