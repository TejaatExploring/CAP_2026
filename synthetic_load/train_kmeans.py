import pandas as pd
import numpy as np
import os
import joblib
import json
from sklearn.cluster import KMeans

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

DATA_PATH = os.path.join(PROJECT_ROOT, "outputs", "cleaned_hourly.csv")
OUT_NPY = os.path.join(PROJECT_ROOT, "outputs", "daily_profiles.npy")

MODEL_PATH = os.path.join(os.path.dirname(__file__), "kmeans_model.pkl")
META_PATH = os.path.join(os.path.dirname(__file__), "metadata.json")

df = pd.read_csv(DATA_PATH, index_col=0, parse_dates=True)

# Ensure full days only
values = df.values.flatten()
days = len(values) // 24
values = values[:days * 24]

daily_profiles = values.reshape(days, 24)
np.save(OUT_NPY, daily_profiles)

# Train KMeans
K = 6
kmeans = KMeans(n_clusters=K, random_state=42, n_init=10)
kmeans.fit(daily_profiles)

joblib.dump(kmeans, MODEL_PATH)

with open(META_PATH, "w") as f:
    json.dump({"clusters": K}, f)

print("✅ KMeans trained")
print("Days:", days)
