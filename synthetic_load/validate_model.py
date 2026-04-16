import numpy as np
import pandas as pd
import os
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error

# =========================================================
# PATH SETUP (ROBUST FOR WINDOWS / VS CODE / JUPYTER)
# =========================================================
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

DATA_PATH = os.path.join(PROJECT_ROOT, "outputs", "cleaned_hourly.csv")
KMEANS_PATH = os.path.join(os.path.dirname(__file__), "kmeans_model.pkl")
MARKOV_PATH = os.path.join(os.path.dirname(__file__), "markov_transition.npy")

# =========================================================
# LOAD DATA
# =========================================================
df = pd.read_csv(DATA_PATH)
actual = df["t_kWh"].values[:168]   # First 7 days (168 hours)

kmeans = joblib.load(KMEANS_PATH)
transition = np.load(MARKOV_PATH)

K = kmeans.n_clusters
states = 24 * K

# =========================================================
# MARKOV-BASED SYNTHETIC GENERATION
# =========================================================
synthetic = []

# Random start
cluster = np.random.randint(K)
hour = 0
state = hour * K + cluster

for t in range(168):
    hour = t % 24
    cluster = state % K

    value = kmeans.cluster_centers_[cluster][hour]
    synthetic.append(value)

    state = np.random.choice(states, p=transition[state])

synthetic = np.array(synthetic)

# =========================================================
# STEP 1: GLOBAL ENERGY SCALING (MANDATORY)
# =========================================================
energy_scale = actual.sum() / synthetic.sum()
synthetic = synthetic * energy_scale

# =========================================================
# STEP 2: HYBRID HOURLY MEAN CORRECTION (KEY IMPROVEMENT)
# =========================================================
actual_hourly_mean = np.zeros(24)
synthetic_hourly_mean = np.zeros(24)

for h in range(24):
    actual_hourly_mean[h] = actual[h::24].mean()
    synthetic_hourly_mean[h] = synthetic[h::24].mean()

corrected = []

for t in range(len(synthetic)):
    h = t % 24

    if synthetic_hourly_mean[h] > 0:
        factor = actual_hourly_mean[h] / synthetic_hourly_mean[h]
    else:
        factor = 1.0

    corrected.append(synthetic[t] * factor)

synthetic = np.array(corrected)

# =========================================================
# ERROR METRICS
# =========================================================
mape = mean_absolute_percentage_error(actual, synthetic) * 100
rmse = np.sqrt(mean_squared_error(actual, synthetic))

print("\n===== FINAL ERROR METRICS =====")
print(f"MAPE : {mape:.2f} %")
print(f"RMSE : {rmse:.3f} kWh")

# =========================================================
# VISUALIZATION
# =========================================================
plt.figure(figsize=(13,5))
plt.plot(actual, label="Actual Load", linewidth=2)
plt.plot(synthetic, "--", label="Synthetic Load (Hybrid)", linewidth=2)
plt.xlabel("Hour")
plt.ylabel("Energy (kWh)")
plt.title("Actual vs Synthetic Load (Hybrid Markov Model)")
plt.legend()
plt.grid(alpha=0.3)
plt.show()
