"""Web animation demo server for Brain 1a pipeline.

This app is intentionally separate from core synthetic_load logic.
It orchestrates existing scripts and streams stage updates/logs to a browser UI.
"""

from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles


ROOT = Path(__file__).resolve().parent.parent
STATIC_DIR = Path(__file__).resolve().parent / "static"


@dataclass
class Stage:
    name: str
    cmd: list[str]


app = FastAPI(title="Brain 1a Animation Demo")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


STAGE_EXPLAIN = {
    "Preprocess": "Reads all Dataset CSVs, auto-detects schema, and builds one clean hourly t_kWh time series.",
    "Train KMeans": "Reshapes hourly load into daily 24-point vectors and learns day-type clusters.",
    "Build Markov": "Learns day-to-day transition probabilities between cluster labels.",
    "Validate": "Runs leakage-free holdout evaluation and compares synthetic output against baseline.",
    "Generate": "Generates final hourly profile and scales energy to the target consumption.",
}


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
def health() -> JSONResponse:
    return JSONResponse({"status": "ok"})


def build_stages(payload: dict[str, Any]) -> list[Stage]:
    days = int(payload.get("days", 30))
    target_kwh = float(payload.get("targetKwh", 350.0))
    target_mode = str(payload.get("targetMode", "monthly"))
    k = int(payload.get("k", 9))
    seed = int(payload.get("seed", 42))
    combine_mode = str(payload.get("combineMode", "mean"))
    chunksize = int(payload.get("chunkSize", 300000))
    laplace = float(payload.get("laplace", 1.0))
    train_ratio = float(payload.get("trainRatio", 0.8))
    noise_scale = float(payload.get("noiseScale", 0.03))
    blend_alpha = float(payload.get("blendAlpha", 0.05))
    skip_train = bool(payload.get("skipTrain", False))

    stages = [
        Stage(
            "Preprocess",
            [
                "python3",
                "synthetic_load/data_preprocessing.py",
                "--combine-mode",
                combine_mode,
                "--chunksize",
                str(chunksize),
            ],
        ),
        Stage(
            "Train KMeans",
            [
                "python3",
                "synthetic_load/train_kmeans.py",
                "--k",
                str(k),
                "--seed",
                str(seed),
            ],
        ),
        Stage(
            "Build Markov",
            ["python3", "synthetic_load/markov_model.py", "--laplace", str(laplace)],
        ),
        Stage(
            "Validate",
            [
                "python3",
                "synthetic_load/validate_model.py",
                "--train-ratio",
                str(train_ratio),
                "--seed",
                str(seed),
                "--no-plot",
            ],
        ),
        Stage(
            "Generate",
            [
                "python3",
                "synthetic_load/generate_synthetic.py",
                "--days",
                str(days),
                "--target-kwh",
                str(target_kwh),
                "--target-mode",
                target_mode,
                "--seed",
                str(seed),
                "--noise-scale",
                str(noise_scale),
                "--blend-alpha",
                str(blend_alpha),
                "--output",
                "outputs/synthetic_30d_hourly.csv",
            ],
        ),
    ]
    if skip_train:
        return [stages[0], stages[3], stages[4]]
    return stages


def parse_metrics(log_line: str) -> dict[str, Any] | None:
    pairs = {
        "synthetic_mape": r"Synthetic MAPE:\s*([0-9.]+)",
        "synthetic_rmse": r"Synthetic RMSE:\s*([0-9.]+)",
        "baseline_mape": r"Baseline MAPE\s*:\s*([0-9.]+)",
        "baseline_rmse": r"Baseline RMSE\s*:\s*([0-9.]+)",
        "rows": r"Rows:\s*([0-9]+)",
        "days": r"Days:\s*([0-9]+)",
        "total_kwh": r"Total energy \(kWh\):\s*([0-9.]+)",
    }

    out: dict[str, Any] = {}
    for key, pat in pairs.items():
        m = re.search(pat, log_line)
        if m:
            raw = m.group(1)
            out[key] = int(raw) if raw.isdigit() else float(raw)
    return out or None


def parse_validate_summary(lines: list[str]) -> dict[str, Any]:
    """Parse final validation metrics from stage logs."""
    text = "\n".join(lines)
    pats = {
        "synthetic_mape": r"Synthetic MAPE:\s*([0-9.]+)",
        "synthetic_rmse": r"Synthetic RMSE:\s*([0-9.]+)",
        "baseline_mape": r"Baseline MAPE\s*:\s*([0-9.]+)",
        "baseline_rmse": r"Baseline RMSE\s*:\s*([0-9.]+)",
        "noise": r"Chosen noise scale:\s*([0-9.]+)",
        "blend_alpha": r"Blend alpha \(synthetic weight\):\s*([0-9.]+)",
    }
    out: dict[str, Any] = {}
    for key, pat in pats.items():
        m = re.search(pat, text)
        if m:
            out[key] = float(m.group(1))
    return out


def stage_summary(stage_name: str, lines: list[str]) -> dict[str, Any]:
    """Build richer stage summary payload for UI internals panel."""
    if stage_name == "Preprocess":
        path = ROOT / "outputs" / "cleaned_hourly.csv"
        if not path.exists():
            return {}
        df = pd.read_csv(path)
        if not {"x_Timestamp", "t_kWh"}.issubset(df.columns):
            return {}
        y = pd.to_numeric(df["t_kWh"], errors="coerce").dropna()
        preview = y.tail(24 * 7).round(6).tolist()
        return {
            "rows": int(len(y)),
            "total_kwh": float(y.sum()),
            "mean_kwh": float(y.mean()) if len(y) else 0.0,
            "preview": preview,
            "preview_hours": len(preview),
        }

    if stage_name == "Train KMeans":
        model_path = ROOT / "synthetic_load" / "kmeans_model.pkl"
        if not model_path.exists():
            return {}
        model = joblib.load(model_path)
        centers = np.asarray(model.cluster_centers_)
        return {
            "clusters": int(model.n_clusters),
            "center_min": float(np.min(centers)),
            "center_max": float(np.max(centers)),
            "centers": centers.round(6).tolist(),
        }

    if stage_name == "Build Markov":
        path = ROOT / "synthetic_load" / "markov_transition.npy"
        if not path.exists():
            return {}
        mat = np.load(path)
        row_sums = mat.sum(axis=1)
        return {
            "shape": [int(mat.shape[0]), int(mat.shape[1])],
            "row_sum_mean": float(np.mean(row_sums)),
            "row_sum_std": float(np.std(row_sums)),
            "matrix": mat.round(5).tolist(),
        }

    if stage_name == "Validate":
        return parse_validate_summary(lines)

    if stage_name == "Generate":
        path = ROOT / "outputs" / "synthetic_30d_hourly.csv"
        if not path.exists():
            return {}
        df = pd.read_csv(path)
        out: dict[str, Any] = {
            "rows": int(len(df)),
            "columns": list(df.columns),
        }
        if "day" in df.columns:
            out["days"] = int(df["day"].nunique())
        if "t_kWh" in df.columns:
            y = pd.to_numeric(df["t_kWh"], errors="coerce").dropna()
            out["total_kwh"] = float(y.sum())
            out["mean_kwh"] = float(y.mean()) if len(y) else 0.0
            out["preview"] = y.head(24 * 7).round(6).tolist()
            out["preview_hours"] = len(out["preview"])
        return out

    return {}


async def send_json(ws: WebSocket, payload: dict[str, Any]) -> None:
    await ws.send_text(json.dumps(payload))


async def run_stage(ws: WebSocket, idx: int, total: int, stage: Stage) -> bool:
    await send_json(
        ws,
        {
            "type": "stage_start",
            "stage": stage.name,
            "index": idx,
            "total": total,
            "command": " ".join(stage.cmd),
            "explain": STAGE_EXPLAIN.get(stage.name, ""),
        },
    )

    proc = await asyncio.create_subprocess_exec(
        *stage.cmd,
        cwd=str(ROOT),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )

    lines: list[str] = []
    assert proc.stdout is not None
    while True:
        line = await proc.stdout.readline()
        if not line:
            break
        text = line.decode("utf-8", errors="replace").rstrip("\n")
        lines.append(text)
        await send_json(ws, {"type": "log", "stage": stage.name, "line": text})

        metric = parse_metrics(text)
        if metric:
            await send_json(ws, {"type": "metric", "stage": stage.name, "data": metric})

    code = await proc.wait()
    ok = code == 0
    summary = stage_summary(stage.name, lines)
    await send_json(
        ws,
        {
            "type": "stage_done",
            "stage": stage.name,
            "index": idx,
            "ok": ok,
            "exitCode": code,
            "summary": summary,
        },
    )
    return ok


@app.websocket("/ws/run")
async def ws_run(ws: WebSocket) -> None:
    await ws.accept()
    try:
        raw = await ws.receive_text()
        payload = json.loads(raw) if raw.strip() else {}
        stages = build_stages(payload)

        await send_json(
            ws,
            {
                "type": "pipeline_start",
                "stages": [s.name for s in stages],
                "config": payload,
            },
        )

        for i, stage in enumerate(stages, start=1):
            ok = await run_stage(ws, i, len(stages), stage)
            if not ok:
                await send_json(ws, {"type": "pipeline_done", "ok": False})
                return

        await send_json(ws, {"type": "pipeline_done", "ok": True})
    except WebSocketDisconnect:
        return
    except Exception as exc:  # pragma: no cover - demo fallback
        await send_json(ws, {"type": "error", "message": str(exc)})
