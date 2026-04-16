# economics.py
# This module handles the economic calculations for the simulation engine.

def calculate_cost(grid_energy_kwh, tariff_per_kwh):
    """Calculate the cost of grid energy consumption."""
    return grid_energy_kwh * tariff_per_kwh
