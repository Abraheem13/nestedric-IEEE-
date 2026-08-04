"""Config resolution, seed sweeps, run directories and result serialisation.

Every run writes a self-describing directory: the resolved config, the normalisation
constants, the environment table with its trace ids, the full T x T matrices and the
metrics. Day 12 needs to recompute statistics without re-running, and Day 15 needs a
fresh clone to reproduce a cell -- both are impossible if a run only leaves a number.
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
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


def _git_commit() -> str:
    """The commit that produced a run, recorded in its results.json.

    A results directory that silently mixes code versions is how a fixed bug appears to
    persist -- or worse, how a broken run gets reported as fixed. Day 7's re-run was
    read from stale files and the numbers came back bit-identical, which looked like the
    fix having no effect.
    """
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            cwd=Path(__file__).resolve().parents[3],
        )
        commit = out.stdout.strip() or "unknown"
        dirty = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            timeout=5,
            cwd=Path(__file__).resolve().parents[3],
        )
        return f"{commit}-dirty" if dirty.stdout.strip() else commit
    except Exception:  # pragma: no cover - git absent or not a repo
        return "unknown"


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


class RunInProgress(RuntimeError):
    """Another process is already writing this run directory."""


def _pid_alive(pid: int) -> bool:
    """Whether a process with *pid* still exists (POSIX; signal 0 tests existence)."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # exists but is not ours
    return True


def _claim(out_dir: Path, force: bool = False) -> Path:
    """Create the run directory now and mark it in progress.

    Two things this buys, both learned the hard way. The directory exists while the run
    is going, so `find ... -name results.json | wc -l` reflects work started rather than
    only work finished -- otherwise a running job looks like a dead one. And a second
    process pointed at the same directory fails immediately instead of interleaving its
    output with the first, which is how two concurrent sweeps quietly corrupt each
    other's results.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    lock = out_dir / ".running"
    if lock.exists() and not force:
        holder = lock.read_text().strip()
        # A crashed run leaves its lock behind. Without this check the directory is
        # poisoned permanently and every later attempt fails with a pid that has not
        # existed for hours -- which is exactly what happened to the first ablation
        # sweep after it died on a ValueError.
        if holder.isdigit() and _pid_alive(int(holder)):
            raise RunInProgress(
                f"{out_dir} is already being written by pid {holder}, which is still "
                "running. Stop that process, or delete the directory to start over."
            )
        print(f"    reclaiming stale lock from pid {holder or '?'} (no such process)")
    lock.write_text(f"{os.getpid()}\n")
    return lock


def run_experiment(cfg: dict, out_dir: Path, force: bool = False) -> Path:
    """Run one (method, stream, seed) experiment; write results.json."""
    lock = _claim(out_dir, force)
    try:
        return _run_experiment_locked(cfg, out_dir)
    finally:
        # Released whatever happens. A lock that outlives its process turns one crash
        # into a permanently unusable run directory.
        lock.unlink(missing_ok=True)


def _run_experiment_locked(cfg: dict, out_dir: Path) -> Path:
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

    # Near-RT feasibility (design rule 6): measured on a real batch from the last
    # environment, at batch size 1, which is the shape of an inference in a control loop.
    from nestedric.eval.footprint import measure_footprint, near_rt_feasible

    probe = next(iter(eval_loaders[stream[-1].env_id]), None)
    if probe is not None:
        fp = measure_footprint(method, probe, device=device)
        results["footprint"] = fp
        results["near_rt_feasible"] = near_rt_feasible(fp)
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
            "platform": {
                "python": platform.python_version(),
                "node": platform.node(),
                "git_commit": _git_commit(),
            },
        }
    )

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


def run_ablation(cfg: dict, out_dir: Path) -> list[Path]:
    """Sweep one axis at a time around the default configuration.

    A *one-at-a-time* sweep, not a full grid. Two reasons: the full cross product of the
    Day 10 axes is 864 cells, which does not fit the schedule; and an ablation answers
    "what does this component contribute to the method as configured", which is a
    question about single deviations from the default. Interactions worth reporting get
    their own explicit cells rather than being buried in a grid nobody can read.

    Every cell runs the same code path as the headline runs, differing from the default
    in exactly one key. An ablation that takes a different branch is not an ablation.
    """
    from nestedric.utils.config import apply_config_overrides, load_config

    method_name = cfg.get("method", "nestedric")
    base_method = load_config(f"configs/method/{method_name}.yaml")
    streams = cfg.get("streams") or [cfg["stream"]]
    seeds = cfg.get("seeds", [0])
    grid = cfg.get("grid", {})

    written: list[Path] = []
    for stream_path in streams:
        stream_name = Path(str(stream_path)).stem
        for axis, values in grid.items():
            for value in values:
                try:
                    method_cfg = apply_config_overrides(base_method, {axis: value})
                except Exception as exc:  # a typo in the grid must not run 40 cells first
                    raise KeyError(f"ablation axis {axis!r} is not a key of {method_name}: {exc}")

                label = f"{axis.replace('.', '_')}={value}".replace(" ", "")
                for seed in seeds:
                    run_dir = out_dir / stream_name / label / f"seed{seed}"
                    if (run_dir / "results.json").exists():
                        print(f"    skip (done) {label} seed {seed}")
                        written.append(run_dir / "results.json")
                        continue
                    one = {
                        **cfg,
                        "stream": stream_path,
                        "method": method_cfg,
                        "method_name": method_name,
                        "seed": seed,
                        "ablation_axis": axis,
                        "ablation_value": value,
                    }
                    print(f"==> {stream_name} / {label} / seed {seed}")
                    written.append(run_experiment(one, run_dir))
    return written
