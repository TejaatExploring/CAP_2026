import numpy as np
import os
import joblib
import json

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

DAILY_PATH = os.path.join(PROJECT_ROOT, "outputs", "daily_profiles.npy")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "kmeans_model.pkl")
META_PATH = os.path.join(os.path.dirname(__file__), "metadata.json")
OUT_PATH = os.path.join(os.path.dirname(__file__), "markov_transition.npy")

daily = np.load(DAILY_PATH)
kmeans = joblib.load(MODEL_PATH)

labels = kmeans.predict(daily)

with open(META_PATH) as f:
    K = json.load(f)["clusters"]

states = 24 * K
transition = np.zeros((states, states))

for d in range(len(labels) - 1):
    for h in range(23):
        s1 = h * K + labels[d]
        s2 = (h + 1) * K + labels[d]
        transition[s1, s2] += 1

    s1 = 23 * K + labels[d]
    s2 = labels[d + 1]
    transition[s1, s2] += 1

transition = transition / transition.sum(axis=1, keepdims=True)
transition = np.nan_to_num(transition)

np.save(OUT_PATH, transition)

print("✅ Markov transition matrix saved")
