# Physics Engine (Rule-Based)

This module simulates end-to-end hourly energy flow for:
- Solar PV generation
- Battery charge/discharge
- Grid import/export

It enforces physical and operational constraints with a deterministic rule-based dispatch.

Baseline default sizing is intentionally moderate:
- PV capacity: 2.5 kW
- Battery capacity: 5.0 kWh
- Battery max charge/discharge: 2.0 kW

## What It Models

- PV power from weather irradiance with simple temperature derating
- Battery operational bounds:
  - SOC min and max
  - max charge and discharge power
  - charge and discharge efficiencies
- Grid constraints:
  - max import power
  - max export power
- Hourly energy flow:
  - load demand
  - PV to load
  - battery discharge to load
  - grid import to load
  - PV to battery
  - grid export
  - curtailment

## Main Metrics Produced

- total_load_kwh
- total_pv_gen_kwh
- total_battery_charge_input_kwh
- total_battery_discharge_to_load_kwh
- total_grid_import_kwh
- total_grid_export_kwh
- total_unserved_load_kwh
- grid_dependency_pct
- self_sufficiency_pct
- average_soc_pct
- min_soc_pct
- max_soc_pct

## Inputs

Load CSV supports columns like:
- x_Timestamp
- t_kWh

Weather CSV supports columns like:
- timestamp_utc
- ghi_kwh_m2 (interpreted as W/m^2 from NASA POWER hourly values)
- wind_speed_10m_mps (optional)
- temperature_2m_c (optional)

If timestamps do not overlap, use align mode by order.

## Run

From repo root:

```bash
python3 -m physics_engine.run_simulation \
  --load-csv outputs/synthetic_30d_hourly.csv \
  --weather-csv outputs/weather_hourly_30d.csv \
  --align-mode by_order \
  --out-csv outputs/physics_simulation_hourly.csv \
  --summary-json outputs/physics_simulation_summary.json
```

To fetch 30 days weather data first:

```bash
python3 weather_service/fetch_weather.py \
  --lat 28.3670 --lon 79.4304 \
  --start 2021-11-01 \
  --days 30 \
  --out outputs/weather_hourly_30d.csv
```

## Notes

- This is intentionally rule-based to keep behavior transparent before adding DQN or GA.
- You can later replace only the dispatch logic while keeping all physical constraints unchanged.
- Terminal output is intentionally concise; full details remain in summary JSON output file.
- If grid dependency is extremely low (for example under 5%), check whether PV and battery are oversized relative to monthly load.
