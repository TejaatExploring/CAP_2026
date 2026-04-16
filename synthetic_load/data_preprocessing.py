import pandas as pd
import os

# ---------- PATH SETUP ----------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

DATA_PATH = os.path.join(
    PROJECT_ROOT, "Datasets", "Smart_meter", "bareilly_2021.csv"
)

OUT_PATH = os.path.join(
    PROJECT_ROOT, "outputs", "cleaned_hourly.csv"
)

# ---------- LOAD DATA ----------
df = pd.read_csv(DATA_PATH)

df["x_Timestamp"] = pd.to_datetime(df["x_Timestamp"])
df.set_index("x_Timestamp", inplace=True)

# Convert 3-min kWh → hourly kWh
hourly = df["t_kWh"].resample("H").sum().dropna()

# Save
os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
hourly.to_csv(OUT_PATH)

print("✅ Hourly data saved")
print("Rows:", len(hourly))
