# Brain 1a Web Animation Demo

This folder contains a standalone web interface to visualize the Brain 1a pipeline while it runs.

It is separate from the core synthetic_load implementation.

## What It Shows

- Stage-by-stage live status (Preprocess, Train, Markov, Validate, Generate)
- Real-time logs from subprocess commands
- Live metric extraction (MAPE, RMSE, rows, days, total kWh)
- Interactive run controls (days, target kWh, mode, K, skip-train)
- Stage internals panel with:
	- Preprocess hourly data preview
	- KMeans cluster-center daily profiles
	- Markov transition matrix heatmap
	- Validation metric breakdown
	- Generated output preview and summary

## Run

From project root:

1. Install dependencies:

pip install -r animation_demo_web/requirements.txt

If WebSocket connection fails, install uvicorn standard extras:

pip install "uvicorn[standard]"

2. Start server (choose one based on where you are):

If you are in project root (CAPSTONE_PROJECT):

python3 -m uvicorn animation_demo_web.server:app --reload --port 8000

If you are already inside animation_demo_web folder:

python3 -m uvicorn server:app --reload --port 8000

3. Open in browser:

http://127.0.0.1:8000

## Notes

- The demo orchestrates existing scripts in synthetic_load.
- It does not change core model logic.
- Use Skip train for a faster presentation run.
- A 404 on /ws/run with warning about unsupported upgrade means websocket libs are missing.
