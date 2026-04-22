# Physics Engine Components and Constraints

This document explains all implemented components in physics_engine, their physical and operational constraints, and how hourly energy flow is simulated.

## Scope

The current implementation is a deterministic rule-based engine for:
- Solar PV generation
- Battery charge and discharge
- Grid import and export
- Hourly energy balancing and summary metrics

It is intentionally transparent so optimization or control layers (DQN, GA, etc.) can be added later without changing core physical logic.

## Implemented Modules

- models.py
  - Defines immutable configuration models for solar, battery, grid, and simulation.
- components.py
  - Implements behavior of SolarPVModel, BatteryModel, and GridModel.
- data_adapter.py
  - Loads CSVs, maps input columns, and aligns load/weather streams.
- simulator.py
  - Executes hourly energy dispatch with constraints and produces outputs.
- run_simulation.py
  - CLI entrypoint for running the simulation pipeline.

## Component Details

### 1) Solar PV Component

Implemented class:
- SolarPVModel

Input signals:
- ghi_w_m2: global horizontal irradiance
- wind_mps: wind speed
- ambient_temp_c: ambient temperature

Main behavior:
- Converts irradiance to power scaled by installed PV capacity.
- Applies derating factor.
- Applies temperature correction using a simple NOCT-style cell temperature estimate.
- Includes a wind-cooling adjustment.

Core relation:
- Base power scales approximately with (GHI / 1000) * PV capacity.
- Temperature factor uses:
  - temp_coeff_per_c
  - reference cell temperature
  - estimated cell temperature from ambient + irradiance rise - wind cooling

Constraints:
- Output power is clamped to non-negative values.
- At zero/negative irradiance, output is zero.

Config fields (SolarConfig):
- capacity_kw
- derate_factor
- temp_coeff_per_c
- noct_c
- reference_cell_temp_c
- default_ambient_temp_c
- wind_cooling_coeff_c_per_mps

### 2) Battery Component

Implemented class:
- BatteryModel

State:
- soc_kwh (absolute stored energy)

Main behavior:
- Charges only from available surplus power.
- Discharges only to cover remaining deficit power.

Operational constraints:
- SOC lower bound: soc_min * capacity_kwh
- SOC upper bound: soc_max * capacity_kwh
- Charge power limit: max_charge_kw
- Discharge power limit: max_discharge_kw
- Charge efficiency: charge_efficiency
- Discharge efficiency: discharge_efficiency

Charge logic constraints:
- Cannot exceed surplus power.
- Cannot exceed max charge power.
- Cannot exceed available room in battery after accounting for efficiency.

Discharge logic constraints:
- Cannot exceed deficit power.
- Cannot exceed max discharge power.
- Cannot exceed deliverable power based on available energy above minimum SOC and discharge efficiency.

Config fields (BatteryConfig):
- capacity_kwh
- soc_initial
- soc_min
- soc_max
- max_charge_kw
- max_discharge_kw
- charge_efficiency
- discharge_efficiency

### 3) Grid Component

Implemented class:
- GridModel

Main behavior:
- Imports power to meet unresolved demand.
- Exports power for unresolved surplus.

Constraints:
- Max import limit: max_import_kw
- Max export limit: max_export_kw
- Any import demand above import limit is marked as unmet load.
- Any export request above export limit is marked as curtailed energy.

Config fields (GridConfig):
- max_import_kw
- max_export_kw

### 4) Data Adapter Component

Implemented functions:
- load_load_data
- load_weather_data
- align_inputs

Main behavior:
- Reads load and weather CSV files.
- Maps supported input column variants.
- Normalizes to a common simulation schema.
- Aligns by timestamp or by row order.

Input handling and constraints:
- Missing or non-numeric critical values are dropped.
- If timestamp alignment has no overlap and mode is timestamp, it raises an error.
- If mode is auto and timestamp overlap is empty, falls back to by_order.
- Path resolver supports common project paths and bare filenames.

Additional simulation coverage check:
- The simulator verifies weather coverage is not shorter than load coverage after alignment.
- If weather is shorter, simulation raises an error and stops.

### 5) Simulation Core

Implemented function:
- run_rule_based_simulation

Time step:
- Fixed hourly stepping by default (dt_hours = 1.0).

Per-hour dispatch sequence:
1. Read load and weather for hour t.
2. Compute PV power for hour t.
3. Serve load directly from PV first.
4. If deficit remains, discharge battery to cover it within limits.
5. If deficit still remains, import from grid within import limit.
6. Any remaining deficit beyond grid limit is unserved load.
7. If PV surplus remains after direct load service, charge battery within limits.
8. If surplus still remains, export to grid within export limit.
9. Any remaining surplus beyond export limit is curtailed.
10. Record SOC and all hourly energy-flow terms.

This order enforces a clear operational policy:
- Prioritize local PV use.
- Then battery for deficits.
- Grid only as needed.
- Store surplus before exporting.

## Hourly Outputs

Generated hourly output includes:
- timestamp
- load_kwh
- pv_gen_kwh
- pv_to_load_kwh
- battery_charge_input_kwh
- battery_stored_kwh
- battery_discharge_to_load_kwh
- grid_import_kwh
- grid_export_kwh
- curtailment_kwh
- unserved_load_kwh
- soc_pct
- ghi_w_m2
- wind_mps
- ambient_temp_c

## Summary Metrics

Generated summary includes:
- total_load_kwh
- total_pv_gen_kwh
- total_battery_charge_input_kwh
- total_battery_stored_kwh
- total_battery_discharge_to_load_kwh
- total_grid_import_kwh
- total_grid_export_kwh
- total_curtailment_kwh
- total_unserved_load_kwh
- average_soc_pct
- min_soc_pct
- max_soc_pct
- grid_dependency_pct
- self_sufficiency_pct
- self_consumption_pct

Metric interpretation:
- grid_dependency_pct = total_grid_import_kwh / total_load_kwh * 100
- self_sufficiency_pct = (total_load_kwh - total_grid_import_kwh) / total_load_kwh * 100
- self_consumption_pct = locally used PV / total PV generation * 100

## Baseline Defaults (Current)

Moderate baseline defaults are used to avoid unrealistic near-off-grid outcomes:
- PV capacity: 2.5 kW
- Battery capacity: 5.0 kWh
- Battery charge/discharge power limits: 2.0 kW

These values can be overridden from CLI.

## Current Assumptions and Limitations

What is modeled:
- First-order PV-weather relationship
- First-order battery and grid constraints
- Energy conservation at hourly resolution

What is not yet modeled:
- Detailed inverter efficiency curve and clipping behavior
- Battery degradation (capacity fade, cycle aging)
- Dynamic tariffs and cost optimization
- Forecast uncertainty and control optimization
- Detailed thermal model beyond simplified NOCT-style approximation

## Why this design is useful now

- Transparent and explainable for debugging and reporting.
- Enforces physical and operational bounds from the start.
- Produces clean state/action/outcome traces suitable for future RL/GA integration.
