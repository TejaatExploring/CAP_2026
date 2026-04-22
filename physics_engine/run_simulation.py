from __future__ import annotations

import argparse

from .models import BatteryConfig, GridConfig, SimulationConfig, SolarConfig
from .simulator import run_rule_based_simulation


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Rule-based physics simulation for PV + battery + grid")

    parser.add_argument("--load-csv", type=str, default="outputs/synthetic_30d_hourly.csv", help="Hourly load CSV")
    parser.add_argument("--weather-csv", type=str, default="outputs/weather_hourly_30d.csv", help="Hourly weather CSV")
    parser.add_argument("--out-csv", type=str, default="outputs/physics_simulation_hourly.csv", help="Hourly result output")
    parser.add_argument("--summary-json", type=str, default="outputs/physics_simulation_summary.json", help="Summary output")

    parser.add_argument("--align-mode", type=str, default="auto", choices=["auto", "timestamp", "by_order"])
    parser.add_argument("--dt-hours", type=float, default=1.0)

    parser.add_argument("--pv-capacity-kw", type=float, default=2.5)
    parser.add_argument("--pv-derate", type=float, default=0.88)
    parser.add_argument("--pv-temp-coeff", type=float, default=-0.004)
    parser.add_argument("--pv-noct-c", type=float, default=45.0)
    parser.add_argument("--ambient-default-c", type=float, default=25.0)
    parser.add_argument("--wind-cooling-coeff", type=float, default=0.6)

    parser.add_argument("--battery-capacity-kwh", type=float, default=5.0)
    parser.add_argument("--soc-initial", type=float, default=0.3)
    parser.add_argument("--soc-min", type=float, default=0.1)
    parser.add_argument("--soc-max", type=float, default=0.95)
    parser.add_argument("--max-charge-kw", type=float, default=2.0)
    parser.add_argument("--max-discharge-kw", type=float, default=2.0)
    parser.add_argument("--charge-eff", type=float, default=0.95)
    parser.add_argument("--discharge-eff", type=float, default=0.95)

    parser.add_argument("--grid-max-import-kw", type=float, default=1e9)
    parser.add_argument("--grid-max-export-kw", type=float, default=1e9)

    return parser


def main() -> None:
    args = build_parser().parse_args()

    solar_config = SolarConfig(
        capacity_kw=args.pv_capacity_kw,
        derate_factor=args.pv_derate,
        temp_coeff_per_c=args.pv_temp_coeff,
        noct_c=args.pv_noct_c,
        default_ambient_temp_c=args.ambient_default_c,
        wind_cooling_coeff_c_per_mps=args.wind_cooling_coeff,
    )
    battery_config = BatteryConfig(
        capacity_kwh=args.battery_capacity_kwh,
        soc_initial=args.soc_initial,
        soc_min=args.soc_min,
        soc_max=args.soc_max,
        max_charge_kw=args.max_charge_kw,
        max_discharge_kw=args.max_discharge_kw,
        charge_efficiency=args.charge_eff,
        discharge_efficiency=args.discharge_eff,
    )
    grid_config = GridConfig(
        max_import_kw=args.grid_max_import_kw,
        max_export_kw=args.grid_max_export_kw,
    )
    simulation_config = SimulationConfig(dt_hours=args.dt_hours, align_mode=args.align_mode)

    _, summary = run_rule_based_simulation(
        load_csv=args.load_csv,
        weather_csv=args.weather_csv,
        solar_config=solar_config,
        battery_config=battery_config,
        grid_config=grid_config,
        simulation_config=simulation_config,
        out_csv=args.out_csv,
        summary_json=args.summary_json,
    )

    print("Rule-based physics simulation completed")
    print(f"Hourly output: {args.out_csv}")
    print(f"Summary output: {args.summary_json}")
    print("--- Summary ---")
    print(f"Steps: {summary['steps']}")
    print(f"Total load (kWh): {summary['total_load_kwh']:.3f}")
    print(f"Total PV generation (kWh): {summary['total_pv_gen_kwh']:.3f}")
    print(f"Grid import (kWh): {summary['total_grid_import_kwh']:.3f}")
    print(f"Grid dependency (%): {summary['grid_dependency_pct']:.2f}")
    print(f"Average SOC (%): {summary['average_soc_pct']:.2f}")
    print(f"Unserved load (kWh): {summary['total_unserved_load_kwh']:.3f}")

    if summary["grid_dependency_pct"] < 5.0:
        print(
            "Note: Grid dependency is very low. This usually means PV/battery are oversized "
            "for the load profile, not a simulator bug."
        )


if __name__ == "__main__":
    main()
