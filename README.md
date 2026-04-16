# CAPSTONE_PROJECT

Integrated Energy Management System for:
- synthetic load generation,
- PV + battery + grid simulation,
- scenario comparison and optimization,
- summary report generation.

## 1) Project Structure

```text
Capstone_Project/
	application/
		report_generator.py
		scenario_generator.py
	optimization/
		optimizer.py
	outputs/
		cleaned_hourly.csv
		daily_profiles.npy
		weather_irradiance.csv
		test_simulation.csv
		simulation_results_formatted.txt
	results/
		simulation_outputs.csv
		optimization_results.csv
		report_summary.json
	simulation_engine/
		run_simulation.py
		simulator.py
		load_model.py
		solar_model.py
		battery_model.py
		energy_flow.py
		economics.py
		weather_solar_fetch.py
	synthetic_load/
		data_preprocessing.py
		train_kmeans.py
		markov_model.py
		validate_model.py
		metadata.json
		markov_transition.npy
```

## 2) How the Simulation Works

Main flow:
1. Load profile is read from `outputs/cleaned_hourly.csv`.
2. Solar generation is computed from `outputs/weather_irradiance.csv` (`shortwave_radiation` or `ghi_wm2`) and system size (`solar_kw`).
3. Energy dispatch order per hour:
	 - Solar -> Load
	 - Remaining Solar -> Battery charge
	 - Battery discharge -> Remaining Load
	 - Grid -> Remaining unmet Load
	 - Unused Solar -> Curtailed Solar
4. Hourly values are aggregated into summary metrics.

Core implementation files:
- `simulation_engine/run_simulation.py`: entry-point script.
- `simulation_engine/simulator.py`: orchestrates models and computes summary.
- `simulation_engine/energy_flow.py`: dispatch logic and result tracking.
- `simulation_engine/battery_model.py`: SoC, charge/discharge limits, efficiency.
- `simulation_engine/solar_model.py`: PV generation model.
- `simulation_engine/load_model.py`: load input parser.

## 3) Inputs and Outputs

### Inputs
- `outputs/cleaned_hourly.csv`
	- Expected load column: `t_kWh` or `load_kWh` or `load_kwh`
- `outputs/weather_irradiance.csv`
	- Expected irradiance column: `shortwave_radiation` or `ghi_wm2`
- Runtime parameters (in `simulation_engine/run_simulation.py` or custom script)
	- `solar_kw`
	- `battery_kwh`
	- `battery_charge_kw`
	- `battery_discharge_kw`

### Outputs
- `outputs/test_simulation.csv`
	- Hourly columns: `load, solar_available, solar_used, battery_charge, battery_discharge, curtailed_solar, grid, soc`
- `outputs/simulation_results_formatted.txt`
	- Human-readable hourly table + summary
- `results/optimization_results.csv`
	- Design-space results for different PV/battery sizes
- `results/report_summary.json`
	- Final KPI summary

## 4) Key Metrics Explained

- `Total Load`: Sum of hourly load.
- `Grid Used`: Sum of hourly grid import.
- `Solar Used`: Solar consumed directly by load.
- `Battery Discharge`: Energy delivered from battery to load.
- `Solar Curtailed`: Generated solar that was neither consumed nor stored.
- `Average SoC (%)`: Mean battery state of charge.
- `Grid Dependency (%)`: `(Grid Used / Total Load) * 100`.

## 5) Quick Start

```bash
pip install -r requirements.txt
python -m simulation_engine.run_simulation
```

Optional:
- Fetch weather data into `outputs/weather_irradiance.csv`:

```bash
python -m simulation_engine.weather_solar_fetch
```

- Run optimization:

```bash
python -m optimization.optimizer
```

- Generate report JSON:

```bash
python -m application.report_generator
```

## 6) Synthetic Load Pipeline

Use `synthetic_load/` when you need training or synthetic profile generation:
1. `synthetic_load/data_preprocessing.py`
2. `synthetic_load/train_kmeans.py`
3. `synthetic_load/markov_model.py`
4. `synthetic_load/validate_model.py`

Generated artifacts include:
- `synthetic_load/markov_transition.npy`
- `synthetic_load/metadata.json`
- model artifacts like `synthetic_load/kmeans_model.pkl`

## 7) Recommended Data Management

To avoid losing old experiments:
- Keep training artifacts under `synthetic_load/`.
- Keep simulation run outputs under `outputs/` and `results/`.
- Create dated snapshot folders before major reruns.
- Commit snapshots to Git for reproducibility.

## 8) Notes

- `Datasets/` (if used) is typically ignored in Git because of size.
- Use relative paths from project root when running modules.
- Python module mode (`python -m ...`) is preferred over direct script execution.
