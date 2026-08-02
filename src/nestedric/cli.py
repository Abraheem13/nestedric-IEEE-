"""Command line entry point. ``nestedric <command> --config <path>``.

Every command takes a config file plus optional ``--set key=value`` overrides, so a run
is fully described by files under version control and one recorded command line.
Nothing that affects a number in the paper is passed any other way.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from nestedric.utils.config import ConfigError, load_config


def build_parser() -> argparse.ArgumentParser:
    """Construct the top-level argument parser.

    Sub-commands
    ------------
    prepare   : materialise raw datasets into the canonical KPI parquet shards.
    stream    : build an O-RAN-CL environment stream and print its environment table.
    train     : run one (method, stream, seed) continual-learning experiment.
    evaluate  : recompute metrics from a finished run directory.
    ablate    : sweep one axis of the NestedRIC configuration.
    figures   : regenerate all paper figures and tables from results/.
    """
    parser = argparse.ArgumentParser(prog="nestedric")
    sub = parser.add_subparsers(dest="command", required=True)

    def add(name: str, help_: str) -> argparse.ArgumentParser:
        p = sub.add_parser(name, help=help_)
        p.add_argument("--config", type=Path, required=True, help="YAML config path")
        p.add_argument(
            "--set",
            dest="overrides",
            action="append",
            default=[],
            metavar="KEY=VALUE",
            help="dotted-key override, repeatable",
        )
        return p

    add("prepare", "raw datasets -> canonical parquet shards")

    p_stream = add("stream", "build an environment stream and print it")
    p_stream.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    p_stream.add_argument("--json", action="store_true", help="emit the stream as JSON")

    add("train", "run one continual-learning experiment")
    add("evaluate", "recompute metrics from a run directory")
    add("ablate", "sweep one axis of the NestedRIC configuration")
    add("figures", "regenerate figures and tables")

    return parser


def _cmd_prepare(cfg: dict, args: argparse.Namespace) -> int:
    """Run preparation for every enabled dataset in a ``configs/data/*.yaml``."""
    from nestedric.data import colosseum as C

    out_dir = Path(cfg.get("output_dir", "data/processed"))
    for entry in cfg.get("datasets", []):
        if not entry.get("enabled", True):
            continue
        name = entry["name"]
        if name not in ("coloran", "commag"):
            print(f"!! no adapter for {name!r} yet -- skipping")
            continue
        root = Path(entry["raw_dir"])
        if not root.exists():
            print(f"!! {root} missing -- run scripts/download_data.sh first")
            continue
        print(f"==> preparing {name} from {root}")
        manifest, _ = C.prepare(root, out_dir, dataset=name)
        print(f"    manifest: {manifest}")
    return 0


def _cmd_stream(cfg: dict, args: argparse.Namespace) -> int:
    """Build a stream and print its environment table (or JSON)."""
    from nestedric.data.stream import StreamError, build_stream

    try:
        stream = build_stream(cfg, args.processed_dir)
    except StreamError as exc:
        print(f"!! {exc}", file=sys.stderr)
        return 2

    if args.json:
        payload = {
            "name": stream.name,
            "family": stream.family,
            "seed": stream.seed,
            "environments": [
                {
                    "env_id": e.env_id,
                    "dataset": e.dataset,
                    "context": e.context,
                    "shards": e.shards,
                    "n_rows": e.n_rows,
                    "train_traces": e.train_traces,
                    "eval_traces": e.eval_traces,
                }
                for e in stream
            ],
        }
        print(json.dumps(payload, indent=2, default=str))
        return 0

    print(stream.table())
    dropped = stream.meta.get("dropped_small")
    if dropped:
        print(f"\n  dropped {len(dropped)} environment(s) below env_min_samples: {dropped[:5]}")
    return 0


def _not_yet(day: str):
    """Placeholder dispatcher for sub-commands scheduled for a later day."""

    def _run(cfg: dict, args: argparse.Namespace) -> int:
        raise NotImplementedError(day)

    return _run


COMMANDS = {
    "prepare": _cmd_prepare,
    "stream": _cmd_stream,
    "train": _not_yet("Day 3"),
    "evaluate": _not_yet("Day 4"),
    "ablate": _not_yet("Day 10"),
    "figures": _not_yet("Day 13"),
}


def main(argv: Sequence[str] | None = None) -> int:
    """Parse *argv*, dispatch to the sub-command, and return an exit code."""
    args = build_parser().parse_args(argv)
    try:
        cfg = load_config(args.config, args.overrides)
    except ConfigError as exc:
        print(f"!! {exc}", file=sys.stderr)
        return 2
    return COMMANDS[args.command](cfg, args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
