from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SolarConfig:
    """Solar PV model configuration."""

    capacity_kw: float = 2.5
    derate_factor: float = 0.88
    temp_coeff_per_c: float = -0.004
    noct_c: float = 45.0
    reference_cell_temp_c: float = 25.0
    default_ambient_temp_c: float = 25.0
    wind_cooling_coeff_c_per_mps: float = 0.6


@dataclass(frozen=True)
class BatteryConfig:
    """Battery model configuration with operational bounds."""

    capacity_kwh: float = 5.0
    soc_initial: float = 0.3
    soc_min: float = 0.1
    soc_max: float = 0.95
    max_charge_kw: float = 2.0
    max_discharge_kw: float = 2.0
    charge_efficiency: float = 0.95
    discharge_efficiency: float = 0.95


@dataclass(frozen=True)
class GridConfig:
    """Grid import/export constraints."""

    max_import_kw: float = 1e9
    max_export_kw: float = 1e9


@dataclass(frozen=True)
class SimulationConfig:
    """Global simulation settings."""

    dt_hours: float = 1.0
    align_mode: str = "auto"
