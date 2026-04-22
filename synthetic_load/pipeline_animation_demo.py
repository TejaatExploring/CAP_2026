"""Animated demo runner for Brain 1a pipeline.

This script is intentionally separate from core model code and is meant for team demos.
It executes the existing pipeline commands and visualizes stage progress live.
"""

from __future__ import annotations

import argparse
import os
import queue
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List


@dataclass
class Stage:
    """A pipeline stage for demo execution."""

    name: str
    command: List[str]


class PipelineAnimator:
    """Matplotlib animator for pipeline stage status and logs."""

    def __init__(self, stage_names: list[str], enabled: bool = True) -> None:
        self.enabled = enabled
        self.stage_names = stage_names
        self._start_ts = time.time()
        self._frame = 0

        if not self.enabled:
            self.fig = None
            self.ax_flow = None
            self.ax_logs = None
            return

        import matplotlib.pyplot as plt
        from matplotlib.patches import FancyBboxPatch

        self.plt = plt
        self.FancyBboxPatch = FancyBboxPatch

        self.plt.ion()
        self.fig = self.plt.figure(figsize=(12, 7))
        self.ax_flow = self.fig.add_subplot(2, 1, 1)
        self.ax_logs = self.fig.add_subplot(2, 1, 2)

    def _stage_color(self, status: str, pulse: bool) -> str:
        if status == "done":
            return "#2e7d32"
        if status == "failed":
            return "#c62828"
        if status == "running":
            return "#f9a825" if pulse else "#fbc02d"
        return "#90a4ae"

    def update(
        self,
        statuses: list[str],
        current_stage: int,
        logs: list[str],
        subtitle: str,
    ) -> None:
        """Render one animation frame."""
        if not self.enabled:
            return

        self._frame += 1
        pulse = (self._frame // 3) % 2 == 0

        self.ax_flow.clear()
        self.ax_logs.clear()

        self.ax_flow.set_xlim(0, max(10, len(self.stage_names) * 2.4))
        self.ax_flow.set_ylim(0, 3)
        self.ax_flow.axis("off")

        for i, name in enumerate(self.stage_names):
            x = 0.8 + i * 2.2
            w, h = 1.8, 0.9
            y = 1.4
            color = self._stage_color(statuses[i], pulse and i == current_stage)

            box = self.FancyBboxPatch(
                (x, y),
                w,
                h,
                boxstyle="round,pad=0.02,rounding_size=0.05",
                linewidth=1.5,
                edgecolor="#37474f",
                facecolor=color,
            )
            self.ax_flow.add_patch(box)
            self.ax_flow.text(
                x + w / 2,
                y + h / 2,
                f"{i + 1}. {name}",
                ha="center",
                va="center",
                fontsize=9,
                color="white",
                weight="bold",
            )

            if i < len(self.stage_names) - 1:
                self.ax_flow.annotate(
                    "",
                    xy=(x + w + 0.2, y + h / 2),
                    xytext=(x + w + 0.02, y + h / 2),
                    arrowprops=dict(arrowstyle="->", lw=2, color="#455a64"),
                )

        elapsed = time.time() - self._start_ts
        self.ax_flow.text(
            0.8,
            2.6,
            f"Brain 1a Pipeline Demo | elapsed: {elapsed:0.1f}s | {subtitle}",
            fontsize=11,
            color="#263238",
            weight="bold",
        )

        self.ax_logs.axis("off")
        self.ax_logs.text(
            0.0,
            1.02,
            "Live Logs",
            transform=self.ax_logs.transAxes,
            fontsize=10,
            weight="bold",
            color="#263238",
        )

        max_lines = 16
        for i, line in enumerate(logs[-max_lines:]):
            self.ax_logs.text(
                0.0,
                0.95 - i * 0.06,
                line[:160],
                transform=self.ax_logs.transAxes,
                fontsize=9,
                family="monospace",
                color="#37474f",
            )

        self.fig.tight_layout()
        self.plt.pause(0.05)

    def close(self) -> None:
        """Close interactive mode and keep final frame."""
        if not self.enabled:
            return
        self.plt.ioff()
        self.plt.show()


def stream_process_output(proc: subprocess.Popen[str], out_queue: queue.Queue[str]) -> None:
    """Read process output lines into a thread-safe queue."""
    assert proc.stdout is not None
    for line in proc.stdout:
        out_queue.put(line.rstrip("\n"))


def run_stage(
    stage: Stage,
    cwd: Path,
    logs: list[str],
    animator: PipelineAnimator,
    statuses: list[str],
    stage_idx: int,
) -> bool:
    """Run one stage and update animation/logs in real time."""
    logs.append(f"$ {' '.join(stage.command)}")
    statuses[stage_idx] = "running"
    animator.update(statuses, stage_idx, logs, f"Running: {stage.name}")

    proc = subprocess.Popen(
        stage.command,
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    out_queue: queue.Queue[str] = queue.Queue()
    reader = threading.Thread(target=stream_process_output, args=(proc, out_queue), daemon=True)
    reader.start()

    while proc.poll() is None or not out_queue.empty():
        try:
            while True:
                logs.append(out_queue.get_nowait())
        except queue.Empty:
            pass

        animator.update(statuses, stage_idx, logs, f"Running: {stage.name}")
        time.sleep(0.1)

    code = proc.returncode
    ok = code == 0
    statuses[stage_idx] = "done" if ok else "failed"
    logs.append(f"[stage {'OK' if ok else 'FAILED'}] {stage.name} (exit={code})")
    animator.update(statuses, stage_idx, logs, f"Completed: {stage.name}")
    return ok


def build_stages(args: argparse.Namespace) -> list[Stage]:
    """Create stage list from CLI options."""
    gen_cmd = [
        "python3",
        "synthetic_load/generate_synthetic.py",
        "--days",
        str(args.days),
        "--target-kwh",
        str(args.target_kwh),
        "--target-mode",
        args.target_mode,
        "--seed",
        str(args.seed),
        "--noise-scale",
        str(args.noise_scale),
        "--blend-alpha",
        str(args.blend_alpha),
        "--output",
        args.output,
    ]

    stages = [
        Stage(
            "Preprocess",
            [
                "python3",
                "synthetic_load/data_preprocessing.py",
                "--combine-mode",
                args.combine_mode,
                "--chunksize",
                str(args.chunksize),
            ],
        ),
        Stage(
            "Train KMeans",
            [
                "python3",
                "synthetic_load/train_kmeans.py",
                "--k",
                str(args.k),
                "--seed",
                str(args.seed),
            ],
        ),
        Stage(
            "Build Markov",
            ["python3", "synthetic_load/markov_model.py", "--laplace", str(args.laplace)],
        ),
        Stage(
            "Validate",
            [
                "python3",
                "synthetic_load/validate_model.py",
                "--train-ratio",
                str(args.train_ratio),
                "--seed",
                str(args.seed),
                "--no-plot",
            ],
        ),
        Stage("Generate 30d+", gen_cmd),
    ]

    if args.skip_train:
        stages = [stages[0], stages[3], stages[4]]
    return stages


def parse_args() -> argparse.Namespace:
    """Parse CLI args for demo runner."""
    parser = argparse.ArgumentParser(description="Animated Brain 1a pipeline demo runner")
    parser.add_argument("--days", type=int, default=30, help="Days for final synthetic generation")
    parser.add_argument("--target-kwh", type=float, default=350.0, help="Energy target input")
    parser.add_argument(
        "--target-mode",
        type=str,
        choices=["monthly", "total"],
        default="monthly",
        help="Interpretation of target-kwh for generation stage",
    )
    parser.add_argument("--output", type=str, default="outputs/synthetic_30d_hourly.csv", help="Output CSV path")
    parser.add_argument("--k", type=int, default=9, help="KMeans cluster count")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--laplace", type=float, default=1.0, help="Markov smoothing")
    parser.add_argument("--train-ratio", type=float, default=0.8, help="Validation split ratio")
    parser.add_argument("--noise-scale", type=float, default=0.03, help="Generator noise scale")
    parser.add_argument("--blend-alpha", type=float, default=0.05, help="Generator blend alpha")
    parser.add_argument("--combine-mode", type=str, choices=["mean", "sum"], default="mean")
    parser.add_argument("--chunksize", type=int, default=300000)
    parser.add_argument("--skip-train", action="store_true", help="Skip training + markov stage for faster demo")
    parser.add_argument("--no-gui", action="store_true", help="Terminal-only mode without matplotlib animation")
    return parser.parse_args()


def main() -> int:
    """Run demo stages and animate progress."""
    args = parse_args()
    root = Path(__file__).resolve().parent.parent

    if args.days <= 0:
        print("days must be positive")
        return 2
    if args.target_kwh <= 0:
        print("target-kwh must be positive")
        return 2

    gui_enabled = not args.no_gui
    if gui_enabled and os.environ.get("DISPLAY") is None and sys.platform.startswith("linux"):
        print("DISPLAY not found; switching to --no-gui mode")
        gui_enabled = False

    stages = build_stages(args)
    statuses = ["pending"] * len(stages)
    logs: list[str] = []

    animator = PipelineAnimator([s.name for s in stages], enabled=gui_enabled)
    animator.update(statuses, 0, logs, "Ready")

    print("Starting animated Brain 1a demo...")
    for i, stage in enumerate(stages):
        if not gui_enabled:
            print(f"\n==> Stage {i + 1}/{len(stages)}: {stage.name}")
        ok = run_stage(stage, root, logs, animator, statuses, i)
        if not ok:
            print(f"Pipeline failed at stage: {stage.name}")
            animator.close()
            return 1

    animator.update(statuses, len(stages) - 1, logs, "Pipeline complete")
    print("\nPipeline demo completed successfully.")
    print(f"Output file: {args.output}")
    animator.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
