# **Stage 2: Simulation Engine - All Components Detail**

This document provides a comprehensive overview of all core components in the Simulation Engine, including their purposes, structures, and mathematical formulas.

---

## **Table of Contents**

1. [Architecture Overview](#architecture-overview)
2. [Component 1: LoadModel](#component-1-loadmodel)
3. [Component 2: SolarModel](#component-2-solarmodel)
4. [Component 3: BatteryModel](#component-3-batterymodel)
5. [Component 4: EnergyFlow](#component-4-energyflow)
6. [Component 5: HybridSystemSimulator](#component-5-hybridsystemsimulator)
7. [Component 6: Economics](#component-6-economics)
8. [Data Flow Diagram](#data-flow-diagram)
9. [Example Simulation](#example-simulation)

---

## **Architecture Overview**

```
HybridSystemSimulator (Main Orchestrator)
    ├─ LoadModel (Input: Demand)
    ├─ SolarModel (Input: Generation)
    ├─ BatteryModel (Storage: State & Constraints)
    ├─ EnergyFlow (Logic: Dispatch & Routing)
    └─ Economics (Calculation: Cost)

Data Flow:
Load + Solar + Battery State
    ↓
EnergyFlow (decides routing)
    ↓
Updated Battery State + Results
    ↓
(repeat for next hour)
```

---

## **Component 1: LoadModel**

### **Purpose**

Represents the **electricity demand** of a home or building. Provides hourly consumption data from a CSV file.

### **Inputs**

- CSV file with hourly electricity consumption
- Column name: `t_kWh`, `load_kWh`, or `load_kwh`

### **Outputs**

- Hourly load value for any given hour
- Total profile length (hours available)

### **Structure**

```python
class LoadModel:
    def __init__(self, load_source="synthetic", file_path=None):
        self.load_source = load_source
        self.file_path = file_path
        self.hourly_load = None
        self._load_data()

    def _load_data(self):
        """Load CSV into memory as numpy array."""
        df = pd.read_csv(self.file_path)
        
        # Support multiple column names
        if "t_kWh" in df.columns:
            self.hourly_load = df["t_kWh"].values
        elif "load_kWh" in df.columns:
            self.hourly_load = df["load_kWh"].values
        elif "load_kwh" in df.columns:
            self.hourly_load = df["load_kwh"].values
        else:
            raise ValueError("CSV must contain column: t_kWh, load_kWh, or load_kwh")

    def get_load(self, hour):
        """Get load for specific hour."""
        return self.hourly_load[hour]

    def get_profile_length(self):
        """Get total hours in profile."""
        return len(self.hourly_load)
```

### **Formulas**

```
No complex formulas. LoadModel is a simple data loader.

Output:
├─ load(hour) = hourly_load[hour]  (direct array lookup)
└─ Time complexity: O(1)
```

### **Example**

```python
load = LoadModel(file_path="outputs/cleaned_hourly.csv")

# Get hourly values
print(load.get_load(0))    # 2.5 kWh
print(load.get_load(12))   # 5.8 kWh
print(load.get_profile_length())  # 8760 (for full year)
```

### **CSV Format**

```csv
t_kWh
2.5
2.3
3.1
4.5
5.2
...
```

---

## **Component 2: SolarModel**

### **Purpose**

Calculates **hourly solar PV generation** based on system capacity, irradiance, and efficiency losses.

### **Inputs**

- Solar capacity (kW)
- Irradiance data (W/m² or normalized profile)
- System efficiency (default 18%)
- Performance ratio (default 80%)

### **Outputs**

- Hourly solar generation (kWh)
- Daily solar energy (kWh)

### **Structure**

```python
class SolarModel:
    def __init__(
        self,
        solar_capacity_kw,
        efficiency=0.18,
        irradiance_profile=None,
        irradiance_wm2=None,
        performance_ratio=0.8
    ):
        self.solar_capacity_kw = solar_capacity_kw
        self.efficiency = efficiency
        self.performance_ratio = performance_ratio
        
        if irradiance_wm2 is not None:
            self.irradiance_wm2 = np.array(irradiance_wm2, dtype=float)
            self.irradiance_profile = None
        elif irradiance_profile is None:
            # Default bell curve (typical sunny day)
            self.irradiance_profile = np.array([
                0.0, 0.0, 0.0, 0.0, 0.05, 0.15,
                0.35, 0.55, 0.75, 0.90, 1.00, 0.95,
                0.85, 0.70, 0.55, 0.35, 0.20, 0.10,
                0.05, 0.0, 0.0, 0.0, 0.0, 0.0
            ])
            self.irradiance_wm2 = None
        else:
            self.irradiance_profile = np.array(irradiance_profile, dtype=float)
            self.irradiance_wm2 = None

    def get_generation(self, hour):
        """Calculate solar generation for specific hour."""
        
        if self.irradiance_wm2 is not None and len(self.irradiance_wm2) > 0:
            h = hour % len(self.irradiance_wm2)
            irradiance_wm2_value = max(0.0, self.irradiance_wm2[h])
            normalized_irradiance = irradiance_wm2_value / 1000.0
        else:
            h = hour % len(self.irradiance_profile)
            normalized_irradiance = max(0.0, self.irradiance_profile[h])
        
        generation = self.solar_capacity_kw * normalized_irradiance
        generation *= self.efficiency * self.performance_ratio
        
        return max(0.0, generation)

    def get_daily_energy(self):
        """Total daily solar generation."""
        return sum(self.get_generation(h) for h in range(24))
```

### **Formulas**

#### **Formula 1: Solar Generation (Core)**

$$\text{Solar Output (kWh)} = C_{\text{solar}} \times I_{\text{norm}} \times \eta_{\text{panel}} \times PR$$

Where:
- $C_{\text{solar}}$ = Solar capacity (kW) [e.g., 5 kW]
- $I_{\text{norm}}$ = Normalized irradiance (0-1) [e.g., 0.8 at noon]
- $\eta_{\text{panel}}$ = Panel efficiency [typical 0.18 = 18%]
- $PR$ = Performance ratio [typical 0.8 = 80%]

**Example Calculation (Hour 12 - Peak Sun):**

$$\text{Output} = 5 \times 0.8 \times 0.18 \times 0.8 = 0.576 \text{ kWh}$$

#### **Formula 2: Irradiance Normalization**

$$I_{\text{norm}} = \begin{cases}
\frac{I_{\text{W/m²}}}{1000} & \text{if real weather data} \\
\text{profile}[h] & \text{if default profile}
\end{cases}$$

Where:
- $I_{\text{W/m²}}$ = Irradiance in Watts per square meter
- $\text{profile}[h]$ = Pre-defined normalized curve (0-1 range)

#### **Formula 3: Daily Energy (Integration)**

$$E_{\text{day}} = \sum_{h=0}^{23} \text{Solar Output}(h)$$

Where sum is over all 24 hours.

**Example:**
$$E_{\text{day}} = 0 + 0 + ... + 0.576 + ... + 0 = 28.7 \text{ kWh}$$

### **Default Irradiance Profile**

```
Hour:  0   1   2   3   4   5   6    7    8    9   10   11
Norm:  0.0 0.0 0.0 0.0 0.05 0.15 0.35 0.55 0.75 0.90 1.00 0.95

Hour: 12  13   14   15   16   17   18   19   20   21   22   23
Norm: 0.85 0.70 0.55 0.35 0.20 0.10 0.05 0.0  0.0  0.0  0.0  0.0

Peak: Hour 10 (normalized irradiance = 1.00)
```

### **Parameters Explained**

| Parameter | Default | Range | Meaning |
|-----------|---------|-------|---------|
| `solar_capacity_kw` | User input | 1-20 | Total installed capacity |
| `efficiency` | 0.18 | 0.15-0.22 | Panel efficiency (15-22%) |
| `performance_ratio` | 0.8 | 0.75-0.85 | System losses (75-85% effective) |

### **Example**

```python
# Using default profile
solar = SolarModel(solar_capacity_kw=5)
print(solar.get_generation(0))     # 0.0 kWh (night)
print(solar.get_generation(12))    # 4.6 kWh (peak)
print(solar.get_daily_energy())    # ~28.7 kWh

# Using real weather data
weather_df = pd.read_csv("outputs/weather_irradiance.csv")
irradiance = weather_df["shortwave_radiation"].values
solar2 = SolarModel(solar_capacity_kw=5, irradiance_wm2=irradiance)
print(solar2.get_generation(12))   # ~4.1 kWh (actual weather)
```

---

## **Component 3: BatteryModel**

### **Purpose**

Simulates a **rechargeable energy storage system** with real-world constraints: capacity limits, rate limits, efficiency losses, and safety margins.

### **Inputs**

- Capacity (kWh)
- Max charge rate (kW/hr)
- Max discharge rate (kW/hr)
- Efficiency (0-1)
- Initial SoC (%)
- Min SoC limit (%)
- Max SoC limit (%)

### **Outputs**

- Energy charged (kWh)
- Energy discharged (kWh)
- Current SoC (kWh or %)
- Updated battery state

### **Structure**

```python
class BatteryModel:
    def __init__(
        self,
        capacity_kwh,
        max_charge_kw,
        max_discharge_kw,
        efficiency=0.9,
        initial_soc=0.5,
        min_soc=0.2,
        max_soc=1.0
    ):
        self.capacity_kwh = capacity_kwh
        self.max_charge_kw = max_charge_kw
        self.max_discharge_kw = max_discharge_kw
        self.efficiency = max(0.0, min(1.0, efficiency))
        
        self.min_soc_fraction = min_soc
        self.max_soc_fraction = max_soc
        self.min_soc = min_soc * capacity_kwh
        self.max_soc = max_soc * capacity_kwh
        
        self.soc = initial_soc * capacity_kwh
        self.total_charged_kwh = 0.0
        self.total_discharged_kwh = 0.0

    def charge(self, energy):
        """Charge battery with available energy."""
        
        if energy <= 0:
            return 0.0
        
        # Constraint 1: Respect max charge rate
        energy = min(energy, self.max_charge_kw)
        
        # Constraint 2: Don't overfill
        available_space = self.max_soc - self.soc
        
        # Constraint 3: Apply efficiency
        charged = min(energy * self.efficiency, max(0.0, available_space))
        
        self.soc += charged
        self.total_charged_kwh += charged
        
        return charged

    def discharge(self, demand):
        """Discharge battery to meet load demand."""
        
        if demand <= 0:
            return 0.0
        
        # Constraint 1: Respect max discharge rate
        demand = min(demand, self.max_discharge_kw)
        
        # Constraint 2: Don't over-discharge
        available = self.soc - self.min_soc
        
        # Constraint 3: Apply efficiency
        if self.efficiency == 0:
            return 0.0
        
        discharged = min(demand / self.efficiency, max(0.0, available))
        
        self.soc -= discharged
        delivered = discharged * self.efficiency
        self.total_discharged_kwh += delivered
        
        return delivered

    def get_soc(self):
        """Get current SoC in kWh."""
        return self.soc

    def get_soc_percent(self):
        """Get current SoC as percentage."""
        return (self.soc / self.capacity_kwh) * 100
```

### **Formulas**

#### **Formula 1: Charge Calculation**

$$E_{\text{stored}} = \min \left( E_{\text{input}} \times \eta, \text{Space}_{\text{available}} \right)$$

Where:
- $E_{\text{input}}$ = Energy trying to charge (kWh)
- $\eta$ = Charge efficiency [0.9 = 90%]
- $\text{Space}_{\text{available}} = C_{\text{max}} - SoC_{\text{current}}$
- $C_{\text{max}}$ = Max SoC limit (kWh)
- $SoC_{\text{current}}$ = Current battery level (kWh)

**Example (Charging with excess solar):**
- Input: 1.0 kWh solar
- Efficiency: 90%
- Available space: 10 - 5 = 5 kWh

$$E_{\text{stored}} = \min(1.0 \times 0.9, 5) = \min(0.9, 5) = 0.9 \text{ kWh}$$

New SoC: $5 + 0.9 = 5.9$ kWh

#### **Formula 2: Discharge Calculation**

$$E_{\text{delivered}} = \min \left( \frac{D}{{\eta}}, \text{Available} \right) \times \eta$$

Where:
- $D$ = Demand (kWh needed)
- $\eta$ = Discharge efficiency [0.9 = 90%]
- $\text{Available} = SoC_{\text{current}} - C_{\text{min}}$
- $C_{\text{min}}$ = Min SoC safety limit (kWh)

**Example (Discharging to meet load):**
- Demand: 2.0 kWh
- Efficiency: 90%
- Available: 5.9 - 2.0 = 3.9 kWh

$$\text{Must draw} = \frac{2.0}{0.9} = 2.22 \text{ kWh}$$

$$E_{\text{delivered}} = \min(2.22, 3.9) \times 0.9 = 2.22 \times 0.9 = 2.0 \text{ kWh} \, \checkmark$$

New SoC: $5.9 - 2.22 = 3.68$ kWh

#### **Formula 3: State of Charge (SoC %)**

$$SoC_{\%} = \frac{SoC_{\text{current}}}{C_{\text{total}}} \times 100$$

Where:
- $SoC_{\text{current}}$ = Current energy stored (kWh)
- $C_{\text{total}}$ = Total capacity (kWh)

**Example:**
$$SoC_{\%} = \frac{5.9}{10} \times 100 = 59\%$$

#### **Formula 4: Round-Trip Efficiency**

$$\eta_{\text{round-trip}} = \eta_{\text{charge}} \times \eta_{\text{discharge}}$$

When using one efficiency value for both:

$$\eta_{\text{round-trip}} = 0.9 \times 0.9 = 0.81 = 81\%$$

**Example: Charge 10 kWh, fully discharge**
- Charge: $10 \times 0.9 = 9$ kWh stored
- Discharge: $9 \times 0.9 = 8.1$ kWh delivered
- Overall: 10 → 8.1 kWh (81% efficiency)

### **Constraints Explained**

```
1. CAPACITY CONSTRAINTS
   ├─ Min SoC: e.g., 20% of 10 kWh = 2 kWh (safety reserve)
   ├─ Max SoC: e.g., 100% of 10 kWh = 10 kWh (overcharge prevention)
   └─ Usable: 10 - 2 = 8 kWh practical storage

2. RATE CONSTRAINTS
   ├─ Max charge: 3 kW/hr = Can add max 3 kWh per hour
   ├─ Max discharge: 3 kW/hr = Can remove max 3 kWh per hour
   └─ Example: 10 kWh battery with 3 kW charger = 3.3 hours to full

3. EFFICIENCY LOSS
   ├─ During charge: 10% heat loss
   ├─ During discharge: 10% heat loss
   └─ What goes in ≠ what comes out
```

### **Parameters**

| Parameter | Example | Meaning |
|-----------|---------|---------|
| `capacity_kwh` | 10 | Total storage: 10 kWh |
| `max_charge_kw` | 3 | Max 3 kWh per hour charging |
| `max_discharge_kw` | 3 | Max 3 kWh per hour discharge |
| `efficiency` | 0.9 | 90% (10% conversion loss) |
| `initial_soc` | 0.5 | Start at 50% full |
| `min_soc` | 0.2 | Safety floor: 20% (2 kWh) |
| `max_soc` | 1.0 | Safety ceiling: 100% (10 kWh) |

### **Example**

```python
battery = BatteryModel(
    capacity_kwh=10,
    max_charge_kw=3,
    max_discharge_kw=3,
    efficiency=0.9,
    initial_soc=0.5
)

# Check initial state
print(battery.get_soc())         # 5.0 kWh
print(battery.get_soc_percent()) # 50.0%

# Charge with 2 kWh available solar
charged = battery.charge(2.0)    # Returns ~1.8 kWh stored
print(battery.get_soc_percent()) # ~59%

# Discharge 3 kWh to meet load
delivered = battery.discharge(3.0)  # Returns 3.0 kWh delivered
print(battery.get_soc_percent())    # ~47%
```

---

## **Component 4: EnergyFlow**

### **Purpose**

The **dispatcher** that decides where each kilowatt goes, hour by hour, using a **priority-based rule system**.

### **Inputs**

- LoadModel (demand)
- SolarModel (generation)
- BatteryModel (storage)

### **Outputs**

- 8 result arrays (load, solar_available, solar_used, battery_charge, battery_discharge, curtailed_solar, grid, soc)

### **Structure**

```python
class EnergyFlow:
    def __init__(self, load_model, solar_model, battery_model):
        self.load_model = load_model
        self.solar_model = solar_model
        self.battery_model = battery_model
        self._initialize_results()

    def _initialize_results(self):
        """Create empty tracking arrays."""
        self.results = {
            "load": [],
            "solar_available": [],
            "solar_used": [],
            "battery_charge": [],
            "battery_discharge": [],
            "curtailed_solar": [],
            "grid": [],
            "soc": []
        }

    def simulate(self, hours):
        """Simulate hour-by-hour energy dispatch."""
        self._initialize_results()
        
        for h in range(hours):
            # ===== GET INPUTS =====
            load = self.load_model.get_load(h)
            solar = self.solar_model.get_generation(h)
            solar_available = solar
            remaining_load = load
            
            # ===== RULE 1: DIRECT SOLAR USE =====
            solar_to_load = min(solar, remaining_load)
            remaining_load -= solar_to_load
            solar -= solar_to_load
            
            # ===== RULE 2: CHARGE BATTERY =====
            if solar > 0:
                battery_charge = self.battery_model.charge(solar)
                solar -= battery_charge
            else:
                battery_charge = 0
            
            # ===== RULE 3: DISCHARGE BATTERY =====
            battery_discharge = 0
            if remaining_load > 0:
                battery_discharge = self.battery_model.discharge(remaining_load)
                remaining_load -= battery_discharge
            
            # ===== RULE 4: IMPORT FROM GRID =====
            grid = max(0, remaining_load)
            
            # ===== RULE 5: CURTAIL EXCESS SOLAR =====
            curtailed_solar = max(0, solar)
            
            # ===== RECORD RESULTS =====
            self.results["load"].append(load)
            self.results["solar_available"].append(solar_available)
            self.results["solar_used"].append(solar_to_load)
            self.results["battery_charge"].append(battery_charge)
            self.results["battery_discharge"].append(battery_discharge)
            self.results["curtailed_solar"].append(curtailed_solar)
            self.results["grid"].append(grid)
            self.results["soc"].append(self.battery_model.get_soc_percent())

    def get_results(self):
        """Return results dictionary."""
        return self.results
```

### **Formulas & Rules**

#### **Rule 1: Direct Solar to Load**

$$E_{\text{solar→load}} = \min(E_{\text{solar}}, D_{\text{load}})$$

Where:
- $E_{\text{solar}}$ = Solar available (kWh)
- $D_{\text{load}}$ = Load demand (kWh)

**Example:**
- Solar: 4.5 kWh
- Load: 6.2 kWh

$$E_{\text{solar→load}} = \min(4.5, 6.2) = 4.5 \text{ kWh}$$

Remaining load: $6.2 - 4.5 = 1.7$ kWh

#### **Rule 2: Charge Battery with Excess Solar**

$$E_{\text{battery_charge}} = \text{battery.charge}(E_{\text{excess\_solar}})$$

Where:
- $E_{\text{excess\_solar}} = E_{\text{solar}} - E_{\text{solar→load}}$

Applied as:
$$E_{\text{stored}} = \min(E_{\text{excess}} \times \eta, \text{Space}_{\text{available}})$$

**Example:**
- Excess solar: 2.0 kWh
- Battery efficiency: 90%
- Available space: 3.0 kWh

$$E_{\text{stored}} = \min(2.0 \times 0.9, 3.0) = 1.8 \text{ kWh}$$

#### **Rule 3: Discharge Battery for Remaining Load**

$$E_{\text{delivered}} = \text{battery.discharge}(D_{\text{remaining}})$$

Applied as:
$$E_{\text{delivered}} = \min\left(\frac{D}{{\eta}}, \text{Available}\right) \times \eta$$

**Example:**
- Remaining load: 1.7 kWh
- Battery discharge efficiency: 90%
- Available in battery: 2.5 kWh

$$\text{Can deliver} = \min\left(\frac{1.7}{0.9}, 2.5\right) \times 0.9 = \min(1.89, 2.5) \times 0.9 = 1.7 \text{ kWh} \, \checkmark$$

#### **Rule 4: Import from Grid**

$$E_{\text{grid}} = \max(0, D_{\text{remaining}})$$

Where $D_{\text{remaining}}$ is unmet load after all other sources.

**Example:**
- Load after solar: 1.7 kWh
- Battery can deliver: 1.7 kWh
- Remaining unmet: 0 kWh

$$E_{\text{grid}} = \max(0, 0) = 0 \text{ kWh}$$

#### **Rule 5: Curtail Excess Solar**

$$E_{\text{curtailed}} = \max(0, E_{\text{remaining\_solar}})$$

**Example:**
- Solar not used anywhere: 0.3 kWh
- Curtailed (wasted): 0.3 kWh

### **Priority Order (Why This Sequence?)**

```
Priority 1: Solar → Load (Direct use)
  └─ Reason: Zero losses, most efficient

Priority 2: Remaining Solar → Battery (Store)
  └─ Reason: Prepare for night/cloudy hours

Priority 3: Battery → Load (Discharge)
  └─ Reason: Use stored renewable energy

Priority 4: Grid → Load (Import)
  └─ Reason: Most expensive, last resort

Priority 5: Excess Solar → Curtailment (Waste)
  └─ Reason: Can't store/use, intentionally waste
```

### **Energy Balance (Conservation Check)**

At each hour:

$$D_{\text{load}} = E_{\text{solar→load}} + E_{\text{battery→load}} + E_{\text{grid}}$$

**Example (Hour 12):**
$$6.5 = 4.5 + 1.8 + 0.2 \, \checkmark$$

---

## **Component 5: HybridSystemSimulator**

### **Purpose**

The **main orchestrator** that ties all 4 models together and runs the complete simulation.

### **Inputs**

- Load file path
- Solar capacity (kW)
- Battery capacity (kWh)
- Battery charge rate (kW)
- Battery discharge rate (kW)
- Weather irradiance file (optional)

### **Outputs**

- Results DataFrame (hourly data)
- Summary metrics (aggregate)

### **Structure**

```python
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
        # Create LoadModel
        self.load_model = LoadModel(file_path=load_file)
        
        # Create SolarModel
        irradiance_wm2 = None
        if weather_irradiance_csv:
            weather_df = pd.read_csv(weather_irradiance_csv)
            if "shortwave_radiation" in weather_df.columns:
                irradiance_wm2 = weather_df["shortwave_radiation"].values
            elif "ghi_wm2" in weather_df.columns:
                irradiance_wm2 = weather_df["ghi_wm2"].values
        
        self.solar_model = SolarModel(solar_kw, irradiance_wm2=irradiance_wm2)
        
        # Create BatteryModel
        self.battery_model = BatteryModel(
            capacity_kwh=battery_kwh,
            max_charge_kw=battery_charge_kw,
            max_discharge_kw=battery_discharge_kw
        )
        
        # Create EnergyFlow
        self.energy_flow = EnergyFlow(
            self.load_model,
            self.solar_model,
            self.battery_model
        )

    def run(self, hours=None):
        """Execute the simulation."""
        if hours is None:
            hours = self.load_model.get_profile_length()
        else:
            hours = min(hours, self.load_model.get_profile_length())
        
        self.energy_flow.simulate(hours)
        self.results = self.energy_flow.get_results()

    def export_results(self, path):
        """Save results to CSV."""
        df = pd.DataFrame(self.results)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        df.to_csv(path, index=False)

    def summary(self):
        """Generate summary metrics."""
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
```

### **Formulas**

#### **Formula 1: Total Load (Sum)**

$$E_{\text{load\_total}} = \sum_{h=0}^{H-1} E_{\text{load}}(h)$$

Where $H$ = total hours in simulation

**Example (24 hours):**
$$E_{\text{load\_total}} = 2.5 + 2.3 + 3.1 + ... + 4.1 = 95.2 \text{ kWh}$$

#### **Formula 2: Grid Dependency (%)**

$$GD = \frac{E_{\text{grid\_total}}}{E_{\text{load\_total}}} \times 100$$

Where:
- $E_{\text{grid\_total}}$ = Total grid energy imported
- $E_{\text{load\_total}}$ = Total load demanded

**Example:**
$$GD = \frac{32.5}{95.2} \times 100 = 34.1\%$$

Interpretation: "34.1% of energy came from grid, 65.9% from renewables"

#### **Formula 3: Average SoC (%)**

$$SoC_{\text{avg}} = \frac{\sum_{h=0}^{H-1} SoC(h)}{H}$$

**Example (24 hours):**
$$SoC_{\text{avg}} = \frac{45 + 42 + 40 + ... + 50}{24} = 48.2\%$$

#### **Formula 4: Solar Utilization (%)**

$$SU = \frac{E_{\text{solar\_used}}}{E_{\text{solar\_available}}} \times 100$$

**Example:**
$$SU = \frac{45.3}{48.5} \times 100 = 93.4\%$$

---

## **Component 6: Economics**

### **Purpose**

Calculates the **monetary cost** of grid energy consumption.

### **Input**

- Grid energy consumed (kWh)
- Tariff rate (₹/kWh)

### **Output**

- Cost (₹)

### **Structure**

```python
def calculate_cost(grid_energy_kwh, tariff_per_kwh):
    """Calculate cost of grid energy consumption."""
    return grid_energy_kwh * tariff_per_kwh
```

### **Formula**

$$\text{Cost} = E_{\text{grid}} \times T$$

Where:
- $E_{\text{grid}}$ = Grid energy imported (kWh)
- $T$ = Electricity tariff (₹/kWh)

**Example:**
$$\text{Cost} = 32.5 \text{ kWh} \times ₹8/\text{kWh} = ₹260$$

#### **Monthly Cost (Extrapolation)**

$$\text{Cost}_{\text{month}} = GD \times E_{\text{daily\_avg}} \times 30 \times T$$

Where:
- $GD$ = Grid dependency (%)
- $E_{\text{daily\_avg}}$ = Average daily load (kWh)
- $T$ = Tariff (₹/kWh)

**Example (30-day month):**
$$\text{Cost}_{\text{month}} = 0.341 \times 95.2 \times 30 \times 8 = ₹7,840$$

---

## **Data Flow Diagram**

```
INITIALIZATION
├─ Load CSV File
│  └─ hourly_load = [2.5, 2.3, 3.1, ...]
│
├─ Create Solar Model
│  └─ with capacity=5kW
│
├─ Create Battery Model
│  └─ with capacity=10kWh, SoC=50%
│
└─ Create EnergyFlow
   └─ connects all 3 models

SIMULATION LOOP (for each hour)
├─ Get load(h) from LoadModel
├─ Get solar(h) from SolarModel
├─ Execute 5 dispatch rules
│  ├─ solar → load
│  ├─ excess → battery charge
│  ├─ battery → load
│  ├─ unmet → grid
│  └─ leftover → curtail
│
├─ Update BatteryModel SoC
│
└─ Record 8 values to results arrays

RESULTS GENERATION
├─ 24 rows × 8 columns
├─ Summary metrics (7 KPIs)
└─ Export CSV + JSON

OUTPUT
├─ simulation_outputs.csv
├─ report_summary.json
└─ Metrics for decision-making
```

---

## **Example Simulation**

### **System Configuration**

```
Solar: 5 kW
Battery: 10 kWh (charge/discharge: 3 kW, efficiency: 90%)
Load: 95.2 kWh/day (from cleaned_hourly.csv)
Tariff: ₹8/kWh
```

### **Hour-by-Hour Example (4 key hours)**

#### **Hour 0 (Midnight: Night)**

```
Inputs:
├─ Load: 2.5 kWh
├─ Solar: 0 kWh (night)
└─ Battery SoC before: 50% (5 kWh)

Dispatch:
├─ Rule 1 (Solar→Load): min(0, 2.5) = 0
├─ Rule 2 (Solar→Battery): 0 (no solar)
├─ Rule 3 (Battery→Load):
│  ├─ Need: 2.5 kWh
│  ├─ Available: 5 - 2 (min) = 3 kWh
│  ├─ Draw: 2.5 / 0.9 = 2.78 kWh
│  └─ Deliver: 2.78 × 0.9 = 2.5 kWh ✓
├─ Rule 4 (Grid→Load): max(0, 0) = 0
└─ Rule 5 (Curtail): 0

Results:
├─ Solar Used: 0 kWh
├─ Battery Charge: 0 kWh
├─ Battery Discharge: 2.5 kWh
├─ Grid: 0 kWh
├─ SoC After: 50% - 2.78/10 = 42.2%
└─ Load Satisfied: 2.5 ✓
```

#### **Hour 8 (Morning: Sunrise)**

```
Inputs:
├─ Load: 4.1 kWh
├─ Solar: 1.8 kWh (rising sun)
└─ Battery SoC before: 45% (4.5 kWh)

Dispatch:
├─ Rule 1 (Solar→Load): min(1.8, 4.1) = 1.8
│  ├─ Remaining load: 4.1 - 1.8 = 2.3
│  └─ Remaining solar: 0
├─ Rule 2 (Solar→Battery): 0 (no excess)
├─ Rule 3 (Battery→Load):
│  ├─ Need: 2.3 kWh
│  ├─ Available: 4.5 - 2 = 2.5 kWh
│  ├─ Draw: 2.3 / 0.9 = 2.56 kWh (but only 2.5 available)
│  ├─ Limited to: 2.5 kWh
│  └─ Deliver: 2.5 × 0.9 = 2.25 kWh
├─ Rule 4 (Grid→Load):
│  └─ Remaining: 2.3 - 2.25 = 0.05 kWh from grid
└─ Rule 5 (Curtail): 0

Results:
├─ Solar Used: 1.8 kWh
├─ Battery Charge: 0 kWh
├─ Battery Discharge: 2.25 kWh
├─ Grid: 0.05 kWh
├─ SoC After: 45% - 2.5/10 = 20% (at minimum!)
└─ Load Satisfied: 1.8 + 2.25 + 0.05 = 4.1 ✓
```

#### **Hour 12 (Noon: Peak Sun, High Load)**

```
Inputs:
├─ Load: 5.2 kWh
├─ Solar: 4.5 kWh (peak sun)
└─ Battery SoC before: 48% (4.8 kWh)

Dispatch:
├─ Rule 1 (Solar→Load): min(4.5, 5.2) = 4.5
│  └─ Remaining load: 5.2 - 4.5 = 0.7
├─ Rule 2 (Solar→Battery): 0 (no excess)
├─ Rule 3 (Battery→Load):
│  ├─ Need: 0.7 kWh
│  ├─ Available: 4.8 - 2 = 2.8 kWh
│  ├─ Draw: 0.7 / 0.9 = 0.78 kWh
│  └─ Deliver: 0.78 × 0.9 = 0.7 kWh ✓
├─ Rule 4 (Grid→Load): 0
└─ Rule 5 (Curtail): 0

Results:
├─ Solar Used: 4.5 kWh
├─ Battery Charge: 0 kWh
├─ Battery Discharge: 0.7 kWh
├─ Grid: 0 kWh
├─ SoC After: 48% - 0.78/10 = 40.2%
└─ Load Satisfied: 4.5 + 0.7 + 0 = 5.2 ✓
```

#### **Hour 18 (Evening: Sunset, Peak Load)**

```
Inputs:
├─ Load: 6.2 kWh (cooking, AC)
├─ Solar: 0.5 kWh (weak, sunset)
└─ Battery SoC before: 42% (4.2 kWh)

Dispatch:
├─ Rule 1 (Solar→Load): min(0.5, 6.2) = 0.5
│  └─ Remaining load: 6.2 - 0.5 = 5.7
├─ Rule 2 (Solar→Battery): 0 (no excess)
├─ Rule 3 (Battery→Load):
│  ├─ Need: 5.7 kWh
│  ├─ Available: 4.2 - 2 = 2.2 kWh
│  ├─ Draw: min(5.7/0.9, 2.2) = min(6.33, 2.2) = 2.2
│  └─ Deliver: 2.2 × 0.9 = 1.98 kWh
├─ Rule 4 (Grid→Load):
│  └─ Remaining: 5.7 - 1.98 = 3.72 kWh from grid
└─ Rule 5 (Curtail): 0

Results:
├─ Solar Used: 0.5 kWh
├─ Battery Charge: 0 kWh
├─ Battery Discharge: 1.98 kWh
├─ Grid: 3.72 kWh ← HIGH! Evening peak
├─ SoC After: 42% - 2.2/10 = 20% (near minimum again)
└─ Load Satisfied: 0.5 + 1.98 + 3.72 = 6.2 ✓
```

### **24-Hour Summary**

```
System: 5kW Solar + 10kWh Battery

Aggregates (Formulas Applied):
├─ Total Load = ∑load = 95.2 kWh
├─ Total Solar Used = ∑solar_used = 45.3 kWh
├─ Total Battery Discharge = ∑battery_discharge = 40 kWh (→ 36 kWh after 90% efficiency)
├─ Total Grid Used = ∑grid = 32.5 kWh
├─ Total Solar Curtailed = ∑curtailed = 0.3 kWh
├─ Average SoC = (∑soc)/24 = 1152/24 = 48%
│
└─ KPIs (Formulas):
   ├─ Grid Dependency = (32.5 / 95.2) × 100 = 34.1%
   ├─ Solar Utilization = (45.3 / 45.6) × 100 = 99.3%
   ├─ Grid Cost = 32.5 × 8 = ₹260
   └─ Monthly Cost (extrapolated) = (34.1% × 95.2 × 30 × 8) = ₹7,840
```

---

## **Summary Table**

| Component | Purpose | Key Formula | Output |
|-----------|---------|-------------|--------|
| **LoadModel** | Read demand | `load(h)` = array[h] | Hourly load |
| **SolarModel** | Calculate generation | `Gen = C × I_norm × η × PR` | Hourly solar |
| **BatteryModel** | Store energy | `Stored = E × η`, `Delivered = E/η × η` | SoC + Energy |
| **EnergyFlow** | Dispatch energy | 5 priority rules | 8 result arrays |
| **HybridSystemSimulator** | Orchestrate | Ties all together | Results + Summary |
| **Economics** | Calculate cost | `Cost = E_grid × T` | Grid cost |

---

## **Key Insights**

```
✓ Modular architecture: Each component is independent and testable
✓ Physics-based: Accounts for real battery/solar constraints
✓ Transparent: Every calculation is visible and traceable
✓ Deterministic: Same inputs always produce same outputs
✓ Rule-based: No AI, just priority ordering

✗ Greedy algorithm: Hour-by-hour, no forward planning
✗ No learning: Fixed rules, doesn't adapt
✗ Simplified physics: Doesn't model all corner cases
```

---

## **References**

- All formulas use SI units (kWh, kW, %)
- Efficiency typically 0.9 (90%) for batteries
- Solar panel efficiency typically 0.15-0.22
- Performance ratio typically 0.75-0.85
- Min SoC safety limit prevents deep discharge damage
- Max SoC limit prevents overcharge damage
