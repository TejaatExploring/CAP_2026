## 🚀 Commands to Run (Step-by-Step)

For new users, here is the recommended order to execute scripts in this folder:

```bash
# 1. Preprocess raw data
python data_preprocessing.py

# 2. Cluster daily profiles
python train_kmeans.py

# 3. Build Markov model and generate synthetic load
python markov_model.py

# 4. Validate synthetic load
python validate_model.py

# 5. (Optional) Explore interactively
jupyter notebook synthetic_load_test.ipynb
```


# Synthetic Load Module

This folder contains scripts, models, and data for generating, validating, and analyzing synthetic electrical load profiles using clustering and Markov models. It is designed to expand limited smart meter datasets for simulation and research.

---

## 📁 Folder Contents

```
synthetic_load/
├── data_preprocessing.py      # Preprocess raw data for modeling
├── markov_model.py            # Markov model for load synthesis
├── train_kmeans.py            # Clustering for load profiles
├── validate_model.py          # Model validation scripts
├── synthetic_load_test.ipynb  # Jupyter notebook for testing/visualization
├── markov_transition.npy      # Markov model transition matrix
├── kmeans_model.pkl           # Trained KMeans clustering model
├── metadata.json              # Metadata for synthetic load
├── load_comparison.png        # Example plot
└── ...
```

---

## 🚦 Workflow: How to Generate Synthetic Load

1. **Preprocess Data**
   - Run `data_preprocessing.py` to convert raw smart meter CSVs (in `../Datasets/Smart_meter/`) into cleaned hourly load data (`../outputs/cleaned_hourly.csv`).
   - This script also creates daily load profiles (`../outputs/daily_profiles.npy`).

2. **Cluster Daily Profiles**
   - Run `train_kmeans.py` to cluster daily profiles using KMeans and save the model (`kmeans_model.pkl`).
   - Metadata (number of clusters, etc.) is saved in `metadata.json`.

3. **Build Markov Model**
   - Run `markov_model.py` to build a Markov transition matrix (`markov_transition.npy`) for hourly load state transitions.
   - This script can also generate synthetic load sequences for any number of days.

4. **Validate Model**
   - Run `validate_model.py` to compare synthetic and real load profiles, compute error metrics (MAPE, RMSE), and visualize results.

5. **Interactive Analysis**
   - Use `synthetic_load_test.ipynb` for step-by-step exploration, visualization, and advanced analysis.

---

## 🧪 Example Scenarios & Model Behavior

### Scenario 1: Generating Synthetic Load for a Typical Month
- **Input:** `monthly_kwh=350`, `days=30`, `random_seed=42`
- **Behavior:** The model generates a synthetic hourly load profile for 30 days, scaled to a total monthly consumption of 350 kWh. The generated profile preserves realistic daily and hourly patterns, including peaks in the morning and evening.
- **Observation:** The synthetic load for the first day (24 hours) might look like:
  `[5.2, 4.6, 4.1, 4.2, 4.5, 4.6, 4.3, 5.7, 11.7, 20.8, 22.3, 19.0, 16.1, 15.4, 15.3, 9.2, 9.4, 12.5, 11.2, 14.5, 14.6, 13.4, 11.1, 6.9]`
- **Plot:** The notebook visualizes the first 7 days, showing realistic daily cycles.

### Scenario 2: Comparing Synthetic and Actual Load
- **Input:** Actual hourly data from `outputs/cleaned_hourly.csv` vs. synthetic load generated for the same period.
- **Behavior:** The notebook aligns the lengths of actual and synthetic data, then plots both for the first 7 days.
- **Observation:** The synthetic load closely follows the actual load's daily shape, but with some differences in magnitude and timing, as expected from a Markov-based model.

### Scenario 3: Error Metrics and Model Validation
- **Input:** Actual and synthetic load arrays.
- **Behavior:** The notebook computes MAPE (Mean Absolute Percentage Error) and RMSE (Root Mean Squared Error) between actual and synthetic loads.
- **Example Output:**
  ```
  MAPE : 8.25 %
  RMSE : 2.13 kWh
  ```
- **Interpretation:** A low MAPE and RMSE indicate the synthetic model is capturing the main patterns of the real data.

### Scenario 4: Visualizing Error Margins
- **Behavior:** The notebook plots the error margin (absolute difference) between actual and synthetic loads, highlighting periods where the model over- or under-estimates demand.

---

## 🏃‍♂️ Commands for Testing & Example Output

To test the synthetic load generation and validation, you can use the provided notebook or run scripts directly from the command line:

```bash
# Generate synthetic load for a month and visualize
jupyter notebook synthetic_load_test.ipynb

# Or run validation script for error metrics and plots
python validate_model.py
```

**Example Output from `validate_model.py`:**

```
===== FINAL ERROR METRICS =====
MAPE : 8.25 %
RMSE : 2.13 kWh
```

This means the synthetic load profile is, on average, within 8.25% of the actual load, with a root mean squared error of 2.13 kWh per hour. The script will also display a plot comparing actual and synthetic loads for visual inspection.

---

## 💡 Example Scenario

Suppose you have 2 months of smart meter data for a neighborhood, but want to simulate a full year for system design. You can:

1. Preprocess the raw data to get clean hourly profiles.
2. Cluster daily patterns to identify typical usage behaviors.
3. Build a Markov model to capture transitions between usage states.
4. Generate synthetic load for 365 days, preserving realistic daily/seasonal patterns.
5. Validate the synthetic data against the real data to ensure accuracy.

---

## 🔄 Extending & Customizing

- Change the number of clusters in `train_kmeans.py` for finer or coarser patterns.
- Modify `markov_model.py` to use different Markov orders or add exogenous variables (e.g., weather).
- Combine multiple CSVs for more robust models.

---

## 📝 Notes

- Outliers (high energy usage) are retained for battery optimization steps.
- Missing values are handled via interpolation or robust methods.
- All outputs are saved in the `outputs/` folder for use in simulation.

---

## 🛠️ Example Usage

```bash
# Step 1: Preprocess data
python data_preprocessing.py

# Step 2: Train KMeans and Markov model
python train_kmeans.py
python markov_model.py

# Step 3: Validate
python validate_model.py
```

---

For questions or improvements, contact the project maintainer.
