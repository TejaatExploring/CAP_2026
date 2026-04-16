
# 🚀 Commands to Run (Step-by-Step)

For new users and developers, here is the recommended order to execute scripts in this folder:

```bash
# 1. (Recommended for development/validation) Test individual modules
python -m simulation_engine.battery_model_test      # Test battery logic
python -m simulation_engine.energy_flow_test        # Test energy flow logic
python -m simulation_engine.solar_model_test        # Test solar model
python -m simulation_engine.load_module_test        # Test load model

# 2. Run the main simulation (24 hours example, uses all modules together)
python -m simulation_engine.simulator_test

# 3. Check results in results/simulation_outputs.csv and review summary in terminal output
```

> **Note:**
> - Run the test files when developing or modifying modules to ensure each part works as expected.
> - For actual simulation studies or scenario analysis, use only the main simulation (simulator_test.py or your own script using HybridSystemSimulator).




# Simulation Engine (Detailed Guide)

This folder contains the core simulation modules for the IEMS project, enabling detailed simulation of hybrid solar-battery-grid systems. It is designed for modularity, extensibility, and educational clarity.

---

## 📁 Folder Contents

```
simulation_engine/
├── battery_model.py         # Battery storage model
├── battery_model_test.py    # Battery model test script
├── economics.py             # Economic calculations (cost, tariff)
├── energy_flow.py           # Energy flow logic (solar, battery, grid)
├── energy_flow_test.py      # Energy flow test script
├── load_model.py            # Load profile model
├── load_module_test.py      # Load model test script
├── simulator.py             # Main simulation runner
├── simulator_test.py        # Example simulation run
├── solar_model.py           # Solar PV generation model
├── solar_model_test.py      # Solar model test script
└── ...
```

---

## 🧩 Module Explanations

### battery_model.py
**BatteryModel**
- Simulates a battery's charging, discharging, and state of charge (SoC).
- Key methods:
	- `charge(energy_kwh)`: Charge the battery, returns energy stored.
	- `discharge(energy_kwh)`: Discharge the battery, returns energy supplied.
	- `get_soc()`, `get_soc_percent()`: Get current SoC in kWh or %.

### energy_flow.py
**EnergyFlow**
- Controls hour-by-hour power flow between solar, battery, load, and grid.
- Key methods:
	- `simulate(hours)`: Run the simulation for a given number of hours.
	- `get_results()`: Retrieve results as a dictionary (load, solar, battery, grid, SoC).

### load_model.py
**LoadModel**
- Loads and provides hourly electrical load data (real or synthetic).
- Key methods:
	- `get_load(hour)`: Get load for a specific hour.
	- `get_total_energy()`: Total energy consumption.
	- `get_peak_load()`: Peak hourly load.

### solar_model.py
**SolarModel**
- Simulates hourly solar PV generation based on capacity and irradiance profile.
- Key methods:
	- `get_generation(hour)`: Get solar generation for a given hour.
	- `get_daily_energy()`: Total daily solar energy.

### simulator.py
**HybridSystemSimulator**
- High-level runner that integrates all models and energy flow.
- Key methods:
	- `run(hours)`: Run the full simulation.
	- `export_results(output_csv)`: Export results to CSV.
	- `summary()`: Return key system metrics (load, grid, solar, SoC).

### economics.py
- `calculate_cost(grid_energy_kwh, tariff_per_kwh)`: Calculate cost of grid energy.

---

## 🚦 Workflow: How to Run a Simulation

1. **Prepare Input Data**
	 - Ensure you have a valid hourly load CSV (e.g., `outputs/cleaned_hourly.csv`).

2. **Test Individual Modules (Recommended for Learning)**
	 - Run test scripts to see how each module works:
		 ```bash
		 python -m simulation_engine.battery_model_test
		 python -m simulation_engine.energy_flow_test
		 python -m simulation_engine.solar_model_test
		 python -m simulation_engine.load_module_test
		 ```

	 - **battery_model_test.py**
		 - Example output:
			 ```
			 Initial SoC: 50.0 %
			 After Charging: 90.0 %
			 After Discharge: 70.0 %
			 ```

	 - **energy_flow_test.py**
		 - Example output:
			 ```
			 Grid Energy (24h): 45.2
			 Final Battery SoC: 48.2 %
			 ```

	 - **solar_model_test.py**
		 - Example output:
			 ```
			 Hour 12 Generation: 4.5
			 Daily Solar Energy: 28.7
			 ```

	 - **load_module_test.py**
		 - Example output:
			 ```
			 Total Energy: 120.5
			 Peak Load: 7.8
			 Hour 10 Load: 4.2
			 ```

3. **Run the Main Simulation**
	 - Use the provided example in `simulator_test.py`:
		 ```bash
		 python -m simulation_engine.simulator_test
		 ```
	 - Or use the `HybridSystemSimulator` class in your own script:
		 ```python
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
		 print(sim.summary())
		 ```

4. **Analyze Results**
	 - Results are saved as CSV (e.g., `results/simulation_outputs.csv`).
	 - Use the summary output for quick system metrics.

---

## 🧪 Example Scenario & Output

**Scenario:** Simulate a 24-hour period for a home with 5 kW solar, 10 kWh battery, and real load data.

**Command:**
```bash
python -m simulation_engine.simulator_test
```

**Example Output:**
```
===== SYSTEM SUMMARY =====
Total Load (kWh): 120.5
Grid Energy (kWh): 45.2
Solar Used (kWh): 60.3
Grid Dependency (%): 37.5
Average Battery SoC (%): 48.2
```

**CSV Output (first few rows):**
```
load,solar,battery_charge,battery_discharge,grid,soc
5.393,0.0,0,3.0,2.393,18.42
4.623,0.0,0,1.75,2.873,0.0
...
```

---

## 🔄 Extending & Customizing

- Add new DER models (e.g., wind, EV) by following the structure of existing modules.
- Modify `simulator.py` to support new control strategies or tariff structures.
- Integrate with the synthetic load module for scenario analysis.

---

## 📝 Notes

- All models are modular and can be tested independently.
- Results can be exported and visualized using pandas, matplotlib, or your preferred tool.
- See the main project README for integration with other modules.
