from simulation_engine.load_model import LoadModel
from simulation_engine.solar_model import SolarModel
from simulation_engine.battery_model import BatteryModel
from simulation_engine.energy_flow import EnergyFlow

load = LoadModel(
    file_path="outputs/cleaned_hourly.csv"
)

solar = SolarModel(
    solar_capacity_kw=5
)

battery = BatteryModel(
    capacity_kwh=10,
    max_charge_kw=3,
    max_discharge_kw=3
)

engine = EnergyFlow(load, solar, battery)
engine.simulate(hours=48)

results = engine.get_results()

print("Grid Energy (24h):", sum(results["grid"]))
print("Final Battery SoC:", results["soc"][-1], "%")
# python -m simulation_engine.energy_flow_test