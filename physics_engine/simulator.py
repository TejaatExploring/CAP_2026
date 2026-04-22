from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from .components import BatteryModel, GridModel, SolarPVModel
from .data_adapter import align_inputs, load_load_data, load_weather_data
from .models import BatteryConfig, GridConfig, SimulationConfig, SolarConfig


def run_rule_based_simulation(
    load_csv: str,
    weather_csv: str,
    solar_config: SolarConfig,
    battery_config: BatteryConfig,
    grid_config: GridConfig,
    simulation_config: SimulationConfig,
    out_csv: str | None = None,
    summary_json: str | None = None,
) -> tuple[pd.DataFrame, dict]:
    load_df = load_load_data(load_csv)
    weather_df = load_weather_data(weather_csv)
    sim_df = align_inputs(load_df, weather_df, mode=simulation_config.align_mode)

    if len(sim_df) < len(load_df):
        raise ValueError(
            "Weather data coverage is shorter than load data. "
            f"Load rows={len(load_df)}, usable weather rows={len(sim_df)}. "
            "Fetch at least matching horizon weather data (for 30 days load, fetch 30 days weather)."
        )

    solar = SolarPVModel(solar_config)
    battery = BatteryModel(battery_config)
    grid = GridModel(grid_config)

    dt = simulation_config.dt_hours

    rows: list[dict] = []

    total_load_kwh = 0.0
    total_pv_gen_kwh = 0.0
    total_pv_to_load_kwh = 0.0
    total_batt_charge_input_kwh = 0.0
    total_batt_stored_kwh = 0.0
    total_batt_discharge_to_load_kwh = 0.0
    total_grid_import_kwh = 0.0
    total_grid_export_kwh = 0.0
    total_curtailment_kwh = 0.0
    total_unserved_kwh = 0.0
    soc_trace: list[float] = []

    for row in sim_df.itertuples(index=False):
        timestamp = row.timestamp
        load_kwh = max(0.0, float(row.load_kwh))
        load_kw = load_kwh / dt if dt > 0 else 0.0

        ambient_temp_c = row.ambient_temp_c
        if pd.isna(ambient_temp_c):
            ambient_temp_c = solar_config.default_ambient_temp_c

        pv_kw = solar.power_kw(
            ghi_w_m2=float(row.ghi_w_m2),
            wind_mps=float(row.wind_mps),
            ambient_temp_c=float(ambient_temp_c),
        )

        pv_to_load_kw = min(pv_kw, load_kw)
        deficit_kw = max(0.0, load_kw - pv_to_load_kw)
        surplus_kw = max(0.0, pv_kw - pv_to_load_kw)

        batt_discharge_kw, _ = battery.discharge_to_cover(deficit_kw=deficit_kw, dt_h=dt)
        remaining_deficit_kw = max(0.0, deficit_kw - batt_discharge_kw)

        grid_import_kw, unmet_kw = grid.import_power(remaining_deficit_kw)

        batt_charge_kw, batt_stored_kwh = battery.charge_from_surplus(surplus_kw=surplus_kw, dt_h=dt)
        remaining_surplus_kw = max(0.0, surplus_kw - batt_charge_kw)

        grid_export_kw, curtailed_kw = grid.export_power(remaining_surplus_kw)

        pv_gen_kwh = pv_kw * dt
        pv_to_load_kwh = pv_to_load_kw * dt
        batt_discharge_to_load_kwh = batt_discharge_kw * dt
        batt_charge_input_kwh = batt_charge_kw * dt
        grid_import_kwh = grid_import_kw * dt
        grid_export_kwh = grid_export_kw * dt
        curtailment_kwh = curtailed_kw * dt
        unserved_kwh = unmet_kw * dt

        soc_pct = battery.soc_pct()
        soc_trace.append(soc_pct)

        total_load_kwh += load_kwh
        total_pv_gen_kwh += pv_gen_kwh
        total_pv_to_load_kwh += pv_to_load_kwh
        total_batt_charge_input_kwh += batt_charge_input_kwh
        total_batt_stored_kwh += batt_stored_kwh
        total_batt_discharge_to_load_kwh += batt_discharge_to_load_kwh
        total_grid_import_kwh += grid_import_kwh
        total_grid_export_kwh += grid_export_kwh
        total_curtailment_kwh += curtailment_kwh
        total_unserved_kwh += unserved_kwh

        rows.append(
            {
                "timestamp": timestamp,
                "load_kwh": load_kwh,
                "pv_gen_kwh": pv_gen_kwh,
                "pv_to_load_kwh": pv_to_load_kwh,
                "battery_charge_input_kwh": batt_charge_input_kwh,
                "battery_stored_kwh": batt_stored_kwh,
                "battery_discharge_to_load_kwh": batt_discharge_to_load_kwh,
                "grid_import_kwh": grid_import_kwh,
                "grid_export_kwh": grid_export_kwh,
                "curtailment_kwh": curtailment_kwh,
                "unserved_load_kwh": unserved_kwh,
                "soc_pct": soc_pct,
                "ghi_w_m2": float(row.ghi_w_m2),
                "wind_mps": float(row.wind_mps),
                "ambient_temp_c": float(ambient_temp_c),
            }
        )

    out_df = pd.DataFrame(rows)

    self_sufficiency_pct = 0.0
    if total_load_kwh > 0:
        self_sufficiency_pct = 100.0 * max(0.0, total_load_kwh - total_grid_import_kwh) / total_load_kwh

    grid_dependency_pct = 0.0
    if total_load_kwh > 0:
        grid_dependency_pct = 100.0 * total_grid_import_kwh / total_load_kwh

    self_consumption_pct = 0.0
    if total_pv_gen_kwh > 0:
        pv_used_locally_kwh = max(0.0, total_pv_gen_kwh - total_grid_export_kwh - total_curtailment_kwh)
        self_consumption_pct = 100.0 * pv_used_locally_kwh / total_pv_gen_kwh
        self_consumption_pct = max(0.0, min(100.0, self_consumption_pct))

    summary = {
        "steps": int(len(out_df)),
        "dt_hours": dt,
        "total_load_kwh": total_load_kwh,
        "total_pv_gen_kwh": total_pv_gen_kwh,
        "total_battery_charge_input_kwh": total_batt_charge_input_kwh,
        "total_battery_stored_kwh": total_batt_stored_kwh,
        "total_battery_discharge_to_load_kwh": total_batt_discharge_to_load_kwh,
        "total_grid_import_kwh": total_grid_import_kwh,
        "total_grid_export_kwh": total_grid_export_kwh,
        "total_curtailment_kwh": total_curtailment_kwh,
        "total_unserved_load_kwh": total_unserved_kwh,
        "average_soc_pct": float(sum(soc_trace) / len(soc_trace)) if soc_trace else 0.0,
        "min_soc_pct": float(min(soc_trace)) if soc_trace else 0.0,
        "max_soc_pct": float(max(soc_trace)) if soc_trace else 0.0,
        "grid_dependency_pct": grid_dependency_pct,
        "self_sufficiency_pct": self_sufficiency_pct,
        "self_consumption_pct": self_consumption_pct,
        "configuration": {
            "solar": asdict(solar_config),
            "battery": asdict(battery_config),
            "grid": asdict(grid_config),
            "simulation": asdict(simulation_config),
        },
    }

    if out_csv:
        out_path = Path(out_csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_df.to_csv(out_path, index=False)

    if summary_json:
        summary_path = Path(summary_json)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    return out_df, summary
