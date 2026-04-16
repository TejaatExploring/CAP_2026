try:
    # Supports: python -m simulation_engine.run_simulation
    from .simulator import HybridSystemSimulator
except ImportError:
    # Supports: python run_simulation.py (from simulation_engine folder)
    from simulator import HybridSystemSimulator

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
WEATHER_FILE = OUTPUTS_DIR / "weather_irradiance.csv"

weather_csv = str(WEATHER_FILE) if WEATHER_FILE.exists() else None

sim = HybridSystemSimulator(
    load_file=str(OUTPUTS_DIR / "cleaned_hourly.csv"),
    solar_kw=3,
    battery_kwh=5,
    battery_charge_kw=2,
    battery_discharge_kw=2,
    weather_irradiance_csv=weather_csv
)

sim.run()

print("\n===== SIMULATION RESULTS =====")
summary = sim.summary()

# Print as aligned table
print("\n{:<30} {:>20}".format("Metric", "Value"))
print("-" * 52)
for key, value in summary.items():
    print("{:<30} {:>20}".format(key, value))

sim.export_results(str(OUTPUTS_DIR / "test_simulation.csv"))

# Also create a formatted table output
import pandas as pd
df_results = pd.DataFrame(sim.results)
formatted_output = str(OUTPUTS_DIR / "simulation_results_formatted.txt")
with open(formatted_output, 'w') as f:
    f.write("===== HOURLY SIMULATION RESULTS =====\n\n")
    f.write(df_results.to_string(index=False))
    f.write("\n\n===== SUMMARY METRICS =====\n")
    for key, value in summary.items():
        f.write(f"{key}: {value}\n")

print("\n✅ Simulation successful!")
print(f"Results saved to: {OUTPUTS_DIR / 'test_simulation.csv'}")
print(f"Formatted results saved to: {formatted_output}")