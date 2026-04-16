from simulation_engine.simulator import HybridSystemSimulator

sim = HybridSystemSimulator(
    load_file="outputs/cleaned_hourly.csv",
    solar_kw=5,
    battery_kwh=10,
    battery_charge_kw=3,
    battery_discharge_kw=3
)

sim.run(hours=24)

sim.export_results("results/simulation_outputs.csv")

summary = sim.summary()

print("===== SYSTEM SUMMARY =====")
for k, v in summary.items():
    print(f"{k}: {v}")
# python -m simulation_engine.simulator_test