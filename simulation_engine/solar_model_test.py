from simulation_engine.solar_model import SolarModel

solar = SolarModel(solar_capacity_kw=5)

print("Hour 12 Generation:", solar.get_generation(12))
print("Daily Solar Energy:", solar.get_daily_energy())
# python -m simulation_engine.solar_model_test