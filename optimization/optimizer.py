# optimization/optimizer.py

import os
import pandas as pd

from simulation_engine.simulator import HybridSystemSimulator
from simulation_engine.economics import calculate_cost


class OptimizationManager:
    """
    Optimization Manager:
    - Generates design space (solar_kw, battery_kwh)
    - Runs simulation for each configuration
    - Selects best based on objective function (cost + grid dependency)
    """

    def __init__(
        self,
        load_file="outputs/cleaned_hourly.csv",
        tariff_per_kwh=8.0,
        results_folder="results"
    ):
        self.load_file = load_file
        self.tariff_per_kwh = tariff_per_kwh
        self.results_folder = results_folder

        os.makedirs(self.results_folder, exist_ok=True)

    def generate_design_space(
        self,
        solar_options=(2, 3, 5, 8, 10),
        battery_options=(5, 10, 15, 20)
    ):
        designs = []
        for s in solar_options:
            for b in battery_options:
                designs.append({"solar_kw": s, "battery_kwh": b})
        return designs

    def objective_function(self, total_grid_kwh, total_load_kwh):
        """
        Objective:
        - Minimize grid dependency + cost
        """
        grid_dependency = (total_grid_kwh / total_load_kwh) * 100 if total_load_kwh > 0 else 100
        cost = calculate_cost(total_grid_kwh, self.tariff_per_kwh)

        # Weighted score (you can tune this)
        score = (0.6 * grid_dependency) + (0.4 * cost)
        return score, grid_dependency, cost

    def run_optimization(self, hours=24):
        designs = self.generate_design_space()
        all_results = []

        print("\n===== OPTIMIZATION STARTED =====")

        for idx, d in enumerate(designs, start=1):
            print(f"\n[{idx}/{len(designs)}] Testing Solar={d['solar_kw']}kW, Battery={d['battery_kwh']}kWh")

            sim = HybridSystemSimulator(
                load_file=self.load_file,
                solar_kw=d["solar_kw"],
                battery_kwh=d["battery_kwh"],
                battery_charge_kw=3,
                battery_discharge_kw=3
            )

            sim.run(hours=hours)
            summary = sim.summary()

            total_load = summary.get("Total Load (kWh)", 0)
            total_grid = summary.get("Grid Energy (kWh)", 0)

            score, grid_dep, cost = self.objective_function(total_grid, total_load)

            all_results.append({
                "solar_kw": d["solar_kw"],
                "battery_kwh": d["battery_kwh"],
                "total_load_kwh": total_load,
                "grid_energy_kwh": total_grid,
                "grid_dependency_percent": grid_dep,
                "grid_cost": cost,
                "score": score
            })

        df = pd.DataFrame(all_results)
        df = df.sort_values("score").reset_index(drop=True)

        output_path = os.path.join(self.results_folder, "optimization_results.csv")
        df.to_csv(output_path, index=False)

        best = df.iloc[0].to_dict()

        print("\n===== OPTIMIZATION COMPLETE =====")
        print("Best Design Found:")
        print(best)
        print(f"\nSaved results -> {output_path}")

        return best, df


if __name__ == "__main__":
    opt = OptimizationManager(
        load_file="outputs/cleaned_hourly.csv",
        tariff_per_kwh=8.0
    )
    best_design, df = opt.run_optimization(hours=24)
