# **CAPSTONE PROJECT - Detailed Explanation**

## **What is This Project About?**

This is an **Energy Management System** that helps you design a solar + battery + grid system for a home or building. It answers: *"What size solar panel and battery should I install to reduce my electricity costs?"*

---

## **The Big Picture - How It Works**

Think of it as a **power flow simulation**:

```
Every Hour:
Solar generates power
    ↓
Powers your home (load)
    ↓
Extra solar charges battery
    ↓
Battery helps when solar is weak
    ↓
Grid covers anything left over
```

The system **tracks all this hourly** and shows you:
- How much electricity you used from the grid
- How much solar you generated and used
- How the battery charged and discharged
- What system size is best for your budget

---

## **Four Main Stages of the Project**

### **Stage 1: Load Data Preparation (synthetic_load/)**

**What it does:** Takes raw electricity meter data and cleans it up.

**Why it matters:** Raw smart-meter data is messy (3-minute readings, gaps, errors). We need clean **hourly data**.

**How it works:**
1. **data_preprocessing.py** → Reads raw CSV → Converts every 3-minute reading into hourly totals → Saves as `cleaned_hourly.csv`
   - Example: Instead of 20 readings/hour, you get 1 number/hour (total kWh used that hour)

2. **train_kmeans.py** → Clusters daily patterns
   - Looks at all the daily load profiles (24-hour patterns)
   - Groups similar days together (e.g., "weekend days look similar", "weekdays look similar")
   - Uses K=6 clusters (6 typical day types)
   - Saves the clustering model

3. **markov_model.py** → Builds a transition model
   - Creates a **Markov state machine** (a model that predicts what comes next based on what's happening now)
   - States: 24 hours × 6 clusters = 144 states
   - Learns: "If we're at hour 10 in cluster 1, what hour+cluster comes next?"
   - Saves transition probabilities

4. **validate_model.py** → Tests if synthetic data matches real data
   - Generates 7 days of fake load data using the Markov model
   - Compares it to the real 7 days
   - Calculates error: MAPE (%) and RMSE (kWh)
   - Plots actual vs synthetic side-by-side
   - Result: "Our synthetic data is 95% accurate" or similar

**Output:** A clean hourly load profile ready for simulation

---

### **Stage 2: Simulation Engine (simulation_engine/)**

**What it does:** Simulates how your solar+battery+grid system works for 24+ hours.

**Core components:**

#### **LoadModel** (load_model.py)
- Reads the hourly load CSV
- Provides load for any given hour
```
Hour 0 → 2.5 kWh
Hour 1 → 2.3 kWh
...
Hour 23 → 4.1 kWh
```

#### **SolarModel** (solar_model.py)
- Takes a solar panel size (e.g., 5 kW) and irradiance data (sunlight strength)
- Calculates hourly solar output
- Accounts for efficiency losses (typical panels are 80% efficient)
```
Example default profile (if no weather data):
Hour 0 → 0 kWh (night)
Hour 6 → 0.3 kWh (sunrise)
Hour 12 → 4.5 kWh (peak sun)
Hour 18 → 0 kWh (sunset)
```

#### **BatteryModel** (battery_model.py)
- Simulates a battery with real constraints:
  - Capacity (e.g., 10 kWh total)
  - Charge/discharge rates (e.g., max 3 kW charge, 3 kW discharge)
  - Efficiency loss (e.g., 90% = 10% loss when charging/discharging)
  - Min/max SoC limits (Storage doesn't run 0–100%, typically 20–100%)
- Tracks state of charge (SoC) percentage

#### **EnergyFlow** (energy_flow.py)
- **The dispatching logic** - decides where power goes each hour

```
Hour-by-hour priority:
1. Load demand = X kWh
2. Solar available = Y kWh
3. Battery current = Z kWh

Step 1: Solar → Load (use as much solar as possible for load)
   used_solar = min(Y, X)
   remaining_load = X - used_solar
   remaining_solar = Y - used_solar

Step 2: Remaining solar → Battery (charge battery with leftover solar)
   battery_charge = battery_model.charge(remaining_solar)
   
Step 3: Battery → Remaining load (discharge if needed)
   battery_discharge = battery_model.discharge(remaining_load)
   remaining_load = remaining_load - battery_discharge

Step 4: Grid → Remaining load (import from grid if still needed)
   grid_import = max(0, remaining_load)

Step 5: Unused solar → Curtailment (throw away excess)
   curtailed = max(0, remaining_solar - battery_charge)
```

#### **HybridSystemSimulator** (simulator.py)
- **Main orchestrator** - stitches all models together
- Runs the simulation for N hours
- Exports hourly results as CSV
- Generates summary metrics

**Output Example (24-hour simulation):**
```
Hour | Load | Solar | Solar→Load | Battery + | Battery - | Grid | SoC
-----|------|-------|-----------|----------|----------|------|-----
0    | 2.5  | 0     | 0         | 0        | 2.5      | 0    | 45%
1    | 2.3  | 0     | 0         | 0        | 2.3      | 0    | 44%
...
12   | 5.0  | 4.5   | 4.5       | 0        | 0        | 0.5  | 52%
...
23   | 3.8  | 0     | 0         | 0        | 3.8      | 0    | 38%

SUMMARY:
Total Load: 95.2 kWh
Solar Used: 45.3 kWh
Battery Discharge: 32.5 kWh
Grid Used: 17.4 kWh
Average SoC: 48%
Grid Dependency: 18.3%
```

---

### **Stage 3: Optimization (optimization/optimizer.py)**

**What it does:** Tries many different solar/battery sizes and finds the **best one**.

**How it works:**

```
Design Space:
Solar sizes: [2, 3, 5, 8, 10] kW
Battery sizes: [5, 10, 15, 20] kWh
Total combinations: 5 × 4 = 20 designs

For each design:
├─ Run simulator
├─ Get metrics (grid used, cost)
├─ Calculate score = (0.6 × grid_dependency) + (0.4 × cost)
└─ Save results

Rank by score (lowest = best)
Return top design
```

**Scoring formula:**
- 60% weight on reducing grid dependency (self-sufficiency)
- 40% weight on minimizing cost (grid electricity price × grid kWh)

**Output:** CSV ranking all 20 designs by score, plus the best one highlighted.

---

### **Stage 4: Application Layer (application/)**

#### **ScenarioGenerator** (scenario_generator.py)
Pre-defined system designs for different budgets:
- **Low Budget:** 3 kW solar, 5 kWh battery (cheap, limited backup)
- **Balanced:** 5 kW solar, 10 kWh battery (good tradeoff)
- **High Resilience:** 8 kW solar, 20 kWh battery (max backup)

#### **ReportGenerator** (report_generator.py)
- Reads simulation CSV
- Calculates summary metrics
- Provides a recommendation:
  ```
  Grid Dependency 70% → "Increase Solar/Battery"
  Grid Dependency 40% → "Balanced mode - OK"
  Grid Dependency 15% → "Solar dominant - Excellent"
  ```
- Exports as JSON for easy viewing

---

## **Data Flow - Complete Picture**

```
RAW DATA
    ↓
[Datasets/Smart_meter/bareilly_2021.csv]
    ↓
synthetic_load/data_preprocessing.py
    ↓
[outputs/cleaned_hourly.csv] (hourly demand)
    ↓
synthetic_load/train_kmeans.py
    ↓
[outputs/daily_profiles.npy] (daily patterns)
[synthetic_load/kmeans_model.pkl] (clustering model)
    ↓
synthetic_load/markov_model.py
    ↓
[synthetic_load/markov_transition.npy] (state transitions)
    ↓
==========================================
READY FOR SIMULATION
==========================================
    ↓
Optional: simulation_engine/weather_solar_fetch.py
    ↓
[outputs/weather_irradiance.csv] (sunlight data)
    ↓
==========================================
    ↓
simulation_engine/run_simulation.py
    ↓
HybridSystemSimulator runs with:
├─ Load: cleaned_hourly.csv
├─ Solar: irradiance CSV (or default pattern)
├─ Battery: 5 kWh, 2 kW charge/discharge
└─ Simulation hours: 24+
    ↓
[outputs/test_simulation.csv] (hourly results)
[outputs/simulation_results_formatted.txt] (human-readable)
    ↓
==========================================
OPTIONAL OPTIMIZATION
==========================================
    ↓
optimization/optimizer.py
    ↓
Tries 20 designs (solar × battery combos)
    ↓
[results/optimization_results.csv] (ranked designs)
    ↓
==========================================
REPORTING
==========================================
    ↓
application/report_generator.py
    ↓
[results/report_summary.json] (summary + recommendation)
```

---

## **Key Metrics Explained**

| Metric | Meaning |
|--------|---------|
| **Total Load** | Total electricity you used (sum of all hours) |
| **Solar Used** | Solar that actually powered your home |
| **Battery Discharge** | Energy pulled from battery to help |
| **Grid Used** | Electricity imported (what you pay for) |
| **Solar Curtailed** | Generated solar that was wasted (too much solar) |
| **Average SoC** | Battery fullness: 50% = half-full |
| **Grid Dependency %** | (Grid Used / Total Load) × 100 = what % from grid |

---

## **Real-World Example**

**Scenario:** Home used 100 kWh over 24 hours

**System Design:** 5 kW solar, 10 kWh battery

**Results:**
```
Hour 0-6 (night):     Load = 15 kWh, Solar = 0 → Battery discharges to 20%
Hour 6-18 (day):      Load = 60 kWh, Solar = 50 kWh → Solar covers load + charges battery
Hour 18-24 (evening): Load = 25 kWh, Solar = 5 kWh → Battery + grid covers rest

Summary:
Total Load = 100 kWh
Solar Used = 45 kWh
Battery Discharge = 30 kWh
Grid Used = 25 kWh ← You only need grid 25% of the time!
Grid Dependency = 25%
```

---

## **Current Issues**

1. **LoadModel missing methods:** The test file expects `get_total_energy()` and `get_peak_load()` but they don't exist in the code.

2. **Optimizer key mismatch:** The optimizer looks for summary keys like `"Total Load (kWh)"` but the simulator returns `"Total Load"`. This means optimizer scoring might calculate zeros.

---

## **Summary**

This is a complete energy system design and optimization platform:
- **Stage 1** cleans and synthesizes realistic load data
- **Stage 2** simulates how different PV/battery sizes perform
- **Stage 3** optimizes to find the best system
- **Stage 4** generates human-friendly reports and recommendations

The goal: Help users make informed decisions about renewable energy investments.
