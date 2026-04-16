import pandas as pd
import os

try:
    from .load_model import LoadModel
    from .solar_model import SolarModel
    from .battery_model import BatteryModel
    from .energy_flow import EnergyFlow
except ImportError:
    from load_model import LoadModel
    from solar_model import SolarModel
    from battery_model import BatteryModel
    from energy_flow import EnergyFlow


class HybridSystemSimulator:
    def __init__(
        self,
        load_file,
        solar_kw,
        battery_kwh,
        battery_charge_kw,
        battery_discharge_kw,
        weather_irradiance_csv=None
    ):
        self.load_model = LoadModel(file_path=load_file)

        irradiance_wm2 = None
        if weather_irradiance_csv:
            weather_df = pd.read_csv(weather_irradiance_csv)
            if "shortwave_radiation" in weather_df.columns:
                irradiance_wm2 = weather_df["shortwave_radiation"].values
            elif "ghi_wm2" in weather_df.columns:
                irradiance_wm2 = weather_df["ghi_wm2"].values

        self.solar_model = SolarModel(solar_kw, irradiance_wm2=irradiance_wm2)

        self.battery_model = BatteryModel(
            capacity_kwh=battery_kwh,
            max_charge_kw=battery_charge_kw,
            max_discharge_kw=battery_discharge_kw
        )

        self.energy_flow = EnergyFlow(
            self.load_model,
            self.solar_model,
            self.battery_model
        )

    def run(self, hours=None):
        if hours is None:
            hours = self.load_model.get_profile_length()
        else:
            hours = min(hours, self.load_model.get_profile_length())

        self.energy_flow.simulate(hours)
        self.results = self.energy_flow.get_results()

    def export_results(self, path):
        df = pd.DataFrame(self.results)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        df.to_csv(path, index=False)

    def summary(self):
        total_load = sum(self.results["load"])
        total_grid = sum(self.results["grid"])
        total_solar = sum(self.results["solar_used"])
        total_battery_discharge = sum(self.results["battery_discharge"])
        total_curtailment = sum(self.results.get("curtailed_solar", []))
        avg_soc = sum(self.results["soc"]) / len(self.results["soc"]) if self.results["soc"] else 0.0

        return {
            "Total Load": float(round(total_load, 2)),
            "Grid Used": float(round(total_grid, 2)),
            "Solar Used": float(round(total_solar, 2)),
            "Battery Discharge": float(round(total_battery_discharge, 2)),
            "Solar Curtailed": float(round(total_curtailment, 2)),
            "Average SoC (%)": float(round(avg_soc, 2)),
            "Grid Dependency (%)": float(round((total_grid / total_load) * 100, 2)) if total_load > 0 else 0.0
        }