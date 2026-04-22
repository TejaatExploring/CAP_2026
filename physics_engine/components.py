from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

from .models import BatteryConfig, GridConfig, SolarConfig


@dataclass
class SolarPVModel:
    config: SolarConfig

    def power_kw(self, ghi_w_m2: float, wind_mps: float, ambient_temp_c: float) -> float:
        """Estimate PV AC-equivalent power with a simple temperature correction."""
        ghi = max(0.0, float(ghi_w_m2))
        wind = max(0.0, float(wind_mps))
        t_amb = float(ambient_temp_c)

        if ghi <= 0.0:
            return 0.0

        # Typical NOCT relation with a lightweight wind-cooling correction.
        temp_rise_c = (self.config.noct_c - 20.0) * (ghi / 800.0)
        cell_temp_c = t_amb + temp_rise_c - (self.config.wind_cooling_coeff_c_per_mps * wind)

        temp_factor = 1.0 + self.config.temp_coeff_per_c * (cell_temp_c - self.config.reference_cell_temp_c)
        temp_factor = max(0.0, temp_factor)

        power = self.config.capacity_kw * (ghi / 1000.0) * self.config.derate_factor * temp_factor
        return max(0.0, power)


@dataclass
class BatteryState:
    soc_kwh: float


class BatteryModel:
    def __init__(self, config: BatteryConfig):
        self.config = config
        initial = max(self.config.soc_min, min(self.config.soc_max, self.config.soc_initial))
        self.state = BatteryState(soc_kwh=initial * self.config.capacity_kwh)

    @property
    def min_soc_kwh(self) -> float:
        return self.config.capacity_kwh * self.config.soc_min

    @property
    def max_soc_kwh(self) -> float:
        return self.config.capacity_kwh * self.config.soc_max

    def soc_pct(self) -> float:
        if self.config.capacity_kwh <= 0:
            return 0.0
        return 100.0 * self.state.soc_kwh / self.config.capacity_kwh

    def charge_from_surplus(self, surplus_kw: float, dt_h: float) -> Tuple[float, float]:
        """Charge from surplus power.

        Returns:
            charge_power_kw: power absorbed from surplus source.
            stored_kwh: actual battery energy increase after charge efficiency.
        """
        if surplus_kw <= 0 or dt_h <= 0:
            return 0.0, 0.0

        room_kwh = max(0.0, self.max_soc_kwh - self.state.soc_kwh)
        if room_kwh <= 0:
            return 0.0, 0.0

        max_power_by_room = room_kwh / (dt_h * max(self.config.charge_efficiency, 1e-9))
        charge_power_kw = min(surplus_kw, self.config.max_charge_kw, max_power_by_room)

        energy_in_kwh = charge_power_kw * dt_h
        stored_kwh = energy_in_kwh * self.config.charge_efficiency
        self.state.soc_kwh += stored_kwh
        self.state.soc_kwh = min(self.state.soc_kwh, self.max_soc_kwh)

        return charge_power_kw, stored_kwh

    def discharge_to_cover(self, deficit_kw: float, dt_h: float) -> Tuple[float, float]:
        """Discharge to serve load deficit.

        Returns:
            discharge_power_kw: power delivered to load.
            extracted_kwh: battery energy removed before discharge efficiency.
        """
        if deficit_kw <= 0 or dt_h <= 0:
            return 0.0, 0.0

        available_kwh = max(0.0, self.state.soc_kwh - self.min_soc_kwh)
        if available_kwh <= 0:
            return 0.0, 0.0

        max_deliverable_kw = (available_kwh * self.config.discharge_efficiency) / dt_h
        discharge_power_kw = min(deficit_kw, self.config.max_discharge_kw, max_deliverable_kw)

        delivered_kwh = discharge_power_kw * dt_h
        extracted_kwh = delivered_kwh / max(self.config.discharge_efficiency, 1e-9)
        self.state.soc_kwh -= extracted_kwh
        self.state.soc_kwh = max(self.state.soc_kwh, self.min_soc_kwh)

        return discharge_power_kw, extracted_kwh


@dataclass
class GridModel:
    config: GridConfig

    def import_power(self, requested_kw: float) -> Tuple[float, float]:
        granted_kw = min(max(0.0, requested_kw), self.config.max_import_kw)
        unmet_kw = max(0.0, requested_kw - granted_kw)
        return granted_kw, unmet_kw

    def export_power(self, requested_kw: float) -> Tuple[float, float]:
        granted_kw = min(max(0.0, requested_kw), self.config.max_export_kw)
        curtailed_kw = max(0.0, requested_kw - granted_kw)
        return granted_kw, curtailed_kw
