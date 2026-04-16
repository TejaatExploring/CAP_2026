import sys
import os
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from simulation_engine.load_model import LoadModel

load_model = LoadModel(
    load_source="synthetic",
    file_path="outputs/cleaned_hourly.csv"
)

print("Total Energy:", load_model.get_total_energy())
print("Peak Load:", load_model.get_peak_load())
print("Hour 10 Load:", load_model.get_load(10))

# python -m simulation_engine.load_module_test