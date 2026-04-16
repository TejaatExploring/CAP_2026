from simulation_engine.battery_model import BatteryModel

battery = BatteryModel(
    capacity_kwh=10,
    max_charge_kw=3,
    max_discharge_kw=3,
    initial_soc=0.5
)

print("Initial SoC:", battery.get_soc_percent(), "%")

battery.charge(4)
print("After Charging:", battery.get_soc_percent(), "%")

battery.discharge(2)
print("After Discharge:", battery.get_soc_percent(), "%")
# python -m simulation_engine.battery_model_test