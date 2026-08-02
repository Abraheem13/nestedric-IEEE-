"""Config resolution, seed sweeps, run directories and result serialisation.

Every run writes a self-describing directory: the resolved config, the normalisation
constants, the environment table with its trace ids, the full T x T matrices and the
metrics. Day 12 needs to recompute statistics without re-running, and Day 15 needs a
fresh clone to reproduce a cell -- both are impossible if a run only leaves a number.
"""

from __future__ import annotations

import json
import platform
import time
from pathlib import Path

from nestedric.data.loaders import build_windows, fit_normaliser
from nestedric.data.schema import FEATURE_COLUMNS
from nestedric.data.stream import build_stream
from nestedric.engine.trainer import ContinualTrainer
from nestedric.eval.evaluator import ContinualEvaluator
from nestedric.methods import build_method
from nestedric.models.backbone import build_backbone
from nestedric.utils.config import load_config
from nestedric.utils.seeding import set_seed


def _device(requested: str = "auto") -> str:
    import torch

    if requested != "auto":
        return requested
    return "cuda" if torch.cuda.is_available() else "cpu"


def _make_loader(ws, batch_size: int, shuffle: bool, seed: int):
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    dataset = TensorDataset(
        torch.from_numpy(ws.x), torch.from_numpy(ws.y), torch.from_numpy(ws.actions)
    )
    generator = torch.Generator().manual_seed(seed) if shuffle else None
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
        drop_last=False,
        num_workers=0,
    )


def run_experiment(cfg: dict, out_dir: Path) -> Path:
    """Run one (method, stream, seed) experiment; write results.json."""
    seed = int(cfg.get("seed", 0))
    set_seed(seed)

    processed = Path(cfg.get("processed_dir", "data/processed"))
    stream_cfg = load_config(cfg["stream"]) if isinstance(cfg["stream"], str) else cfg["stream"]
    model_cfg = load_config(cfg["model"]) if isinstance(cfg["model"], str) else cfg["model"]
    method_cfg = load_config(cfg["method"]) if isinstance(cfg["method"], str) else cfg["method"]

    stream = build_stream(stream_cfg, processed)
    if len(stream) < 2:
        raise RuntimeError(f"stream {stream.name!r} has {len(stream)} environments; need >= 2")

    # Design rule 5: constants come from the source environment only -- the first one
    # the learner sees. Everything afterwards is, by construction, the future.
    n_source = int(cfg.get("n_source_envs", 1))
    normaliser = fit_normaliser(list(stream)[:n_source], processed)

    window = int(model_cfg.get("input", {}).get("window", 32))
    stride = int(cfg.get("stride", 8))
    batch_size = int(method_cfg.get("batch_size", 256))

    train_loaders, eval_loaders, window_counts = {}, {}, {}
    for env in stream:
        tr = build_windows(env, processed, normaliser, env.train_traces, window, stride)
        ev = build_windows(env, processed, normaliser, env.eval_traces, window, stride)
        train_loaders[env.env_id] = _make_loader(tr, batch_size, True, seed)
        eval_loaders[env.env_id] = _make_loader(ev, batch_size, False, seed)
        window_counts[env.env_id] = {"train": len(tr), "eval": len(ev)}

    device = _device(str(cfg.get("device", "auto")))
    model = build_backbone(model_cfg, in_dim=len(FEATURE_COLUMNS) + 1)
    method = build_method(cfg["method_name"], model, {**method_cfg, "seed": seed}, device)

    evaluator = ContinualEvaluator(stream, eval_loaders, device)
    trainer = ContinualTrainer(method, stream, evaluator, train_loaders, method_cfg)

    t0 = time.time()
    results = trainer.run()
    results.update(
        {
            "method": cfg["method_name"],
            "stream": stream.name,
            "seed": seed,
            "device": device,
            "wall_seconds": time.time() - t0,
            "window_counts": window_counts,
            "normaliser": normaliser.to_dict(),
            "environments": [
                {
                    "env_id": e.env_id,
                    "dataset": e.dataset,
                    "context": e.context,
                    "n_rows": e.n_rows,
                    "train_traces": e.train_traces,
                    "eval_traces": e.eval_traces,
                }
                for e in stream
            ],
            "platform": {"python": platform.python_version(), "node": platform.node()},
        }
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.json").write_text(json.dumps(results, indent=2, default=str))
    (out_dir / "config.json").write_text(json.dumps(cfg, indent=2, default=str))
    return out_dir / "results.json"


def run_sweep(cfg: dict, out_dir: Path) -> list[Path]:
    """Run a grid over methods x streams x seeds."""
    streams = cfg.get("streams") or [cfg["stream"]]
    methods = cfg.get("methods") or [cfg["method"]]
    seeds = cfg.get("seeds", [0])

    written: list[Path] = []
    for stream_path in streams:
        for method_name in methods:
            for seed in seeds:
                name = Path(str(stream_path)).stem
                run_dir = out_dir / name / method_name / f"seed{seed}"
                one = {
                    **cfg,
                    "stream": stream_path,
                    "method": f"configs/method/{method_name}.yaml",
                    "method_name": method_name,
                    "seed": seed,
                }
                print(f"==> {name} / {method_name} / seed {seed}")
                written.append(run_experiment(one, run_dir))
    return written
