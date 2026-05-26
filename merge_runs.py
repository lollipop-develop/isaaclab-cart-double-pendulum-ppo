#!/usr/bin/env python
"""Concatenate sequential training runs into a single continuous TensorBoard curve.

Each training run's tensorboard events start at step 0, so when you resume
training (or split a single conceptual training session across multiple runs)
the curves appear separately in tensorboard and you can't read them as one
continuous trajectory.

This script reads the scalars from N runs in chronological order and rewrites
them into a single new run directory, offsetting each successive run's step
numbers by the cumulative max step of the runs before it. The output is a
normal tensorboard run that displays as a single continuous curve.

Usage:
    python merge_runs.py runs/run1 runs/run2 runs/run3 --output runs/merged

The runs MUST be in chronological order — first argument is the earliest.
The output directory is created if it doesn't exist.

Requires:  tensorboard, torch (already installed for training).
Does not require Isaac Sim — runs as plain Python.
"""

from __future__ import annotations

import argparse
import os
import sys

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from torch.utils.tensorboard import SummaryWriter


def load_scalars(run_dir: str) -> dict[str, list[tuple[int, float]]]:
    """Read every scalar event from a single run directory.

    Returns a dict mapping tag name -> list of (step, value) pairs.
    """
    # size_guidance={'scalars': 0} disables tensorboard's default downsampling
    # so we don't drop points on long runs.
    acc = EventAccumulator(run_dir, size_guidance={"scalars": 0})
    acc.Reload()
    scalar_tags = acc.Tags().get("scalars", [])
    return {
        tag: [(e.step, float(e.value)) for e in acc.Scalars(tag)]
        for tag in scalar_tags
    }


def main() -> int:
    p = argparse.ArgumentParser(
        description="Concatenate sequential training runs into a single TensorBoard curve.",
    )
    p.add_argument(
        "runs",
        nargs="+",
        help="Run directories in chronological order (first arg = earliest).",
    )
    p.add_argument(
        "--output",
        required=True,
        help="Output directory for the merged run (created if missing).",
    )
    args = p.parse_args()

    for r in args.runs:
        if not os.path.isdir(r):
            print(f"ERROR: not a directory: {r}", file=sys.stderr)
            return 1

    os.makedirs(args.output, exist_ok=True)
    writer = SummaryWriter(args.output)

    step_offset = 0
    for run_dir in args.runs:
        scalars = load_scalars(run_dir)
        if not scalars:
            print(f"[merge] WARNING: no scalars in {run_dir}, skipping")
            continue
        max_step = 0
        n_points = 0
        for tag, events in scalars.items():
            for step, value in events:
                writer.add_scalar(tag, value, step + step_offset)
                if step > max_step:
                    max_step = step
                n_points += 1
        print(
            f"[merge] {run_dir}: {n_points} points across {len(scalars)} tags, "
            f"steps 0–{max_step} → wrote at {step_offset}–{step_offset + max_step}"
        )
        # Use max_step + 1 so the next run's step 0 doesn't overlap our last step.
        step_offset += max_step + 1

    writer.close()
    print(f"\n[merge] wrote merged run to {args.output}")
    print(f"[merge] view with: tensorboard --logdir {os.path.dirname(args.output) or '.'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
