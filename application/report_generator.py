# application/report_generator.py

import json
import pandas as pd


class ReportGenerator:
    """
    Report Generator:
    Reads simulation output CSV and generates decision-support metrics.
    """

    def __init__(self, simulation_csv="results/simulation_outputs.csv"):
        self.simulation_csv = simulation_csv

    def generate_report(self):
        df = pd.read_csv(self.simulation_csv)

        total_load = df["load"].sum()
        total_solar = df["solar"].sum()
        total_grid = df["grid"].sum()
        total_battery_discharge = df["battery_discharge"].sum()

        grid_dependency = (total_grid / total_load) * 100 if total_load > 0 else 100

        report = {
            "total_load_kwh": round(total_load, 3),
            "total_solar_used_kwh": round(total_solar, 3),
            "total_grid_energy_kwh": round(total_grid, 3),
            "total_battery_discharge_kwh": round(total_battery_discharge, 3),
            "grid_dependency_percent": round(grid_dependency, 2),
            "recommendation": self.get_recommendation(grid_dependency)
        }

        return report

    def get_recommendation(self, grid_dependency):
        if grid_dependency > 70:
            return "Grid Dominant (Increase Solar/Battery recommended)"
        elif grid_dependency > 40:
            return "Hybrid Mode (Balanced Solar + Grid)"
        else:
            return "Solar Dominant (Good renewable utilization)"

    def export_json(self, output_file="results/report_summary.json"):
        report = self.generate_report()

        with open(output_file, "w") as f:
            json.dump(report, f, indent=4)

        print(f"Report saved -> {output_file}")
        return report


if __name__ == "__main__":
    rg = ReportGenerator("results/simulation_outputs.csv")
    print(rg.export_json())
