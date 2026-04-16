import numpy as np
import pandas as pd
import os

class LoadModel:
    def __init__(self, load_source="synthetic", file_path=None):
        self.load_source = load_source
        self.file_path = file_path
        self.hourly_load = None
        self._load_data()

    def _load_data(self):
        if self.file_path is None:
            raise ValueError("file_path must be provided")

        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"Load file not found: {self.file_path}")

        df = pd.read_csv(self.file_path)

        # ✅ FIXED: support all formats
        if "t_kWh" in df.columns:
            self.hourly_load = df["t_kWh"].values
        elif "load_kWh" in df.columns:
            self.hourly_load = df["load_kWh"].values
        elif "load_kwh" in df.columns:
            self.hourly_load = df["load_kwh"].values
        else:
            raise ValueError("CSV must contain 't_kWh' or 'load_kWh' or 'load_kwh'")

    def get_load(self, hour):
        return self.hourly_load[hour]

    def get_profile_length(self):
        return len(self.hourly_load)