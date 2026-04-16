import numpy as np


class SolarModel:
    def __init__(
        self,
        solar_capacity_kw,
        efficiency=0.18,
        irradiance_profile=None,
        irradiance_wm2=None,
        performance_ratio=0.8
    ):
        self.solar_capacity_kw = solar_capacity_kw
        self.efficiency = efficiency
        self.performance_ratio = performance_ratio

        if irradiance_wm2 is not None:
            self.irradiance_wm2 = np.array(irradiance_wm2, dtype=float)
            self.irradiance_profile = None
        elif irradiance_profile is None:
            self.irradiance_profile = np.array([
                0.0, 0.0, 0.0, 0.0, 0.05, 0.15,
                0.35, 0.55, 0.75, 0.90, 1.00, 0.95,
                0.85, 0.70, 0.55, 0.35, 0.20, 0.10,
                0.05, 0.0, 0.0, 0.0, 0.0, 0.0
            ])
            self.irradiance_wm2 = None
        else:
            self.irradiance_profile = np.array(irradiance_profile, dtype=float)
            self.irradiance_wm2 = None

    # --------------------------------------------------
    def get_generation(self, hour):
        """
        Returns solar generation at a given hour (kWh)
        """
        if self.irradiance_wm2 is not None and len(self.irradiance_wm2) > 0:
            h = hour % len(self.irradiance_wm2)
            normalized_irradiance = max(0.0, self.irradiance_wm2[h]) / 1000.0
        else:
            h = hour % len(self.irradiance_profile)
            normalized_irradiance = max(0.0, self.irradiance_profile[h])

        generation = self.solar_capacity_kw * normalized_irradiance
        generation *= self.efficiency * self.performance_ratio

        return max(0.0, generation)

    # --------------------------------------------------
    def get_daily_energy(self):
        span = 24
        if self.irradiance_wm2 is not None and len(self.irradiance_wm2) > 0:
            span = min(24, len(self.irradiance_wm2))
        elif self.irradiance_profile is not None and len(self.irradiance_profile) > 0:
            span = min(24, len(self.irradiance_profile))

        return sum(self.get_generation(h) for h in range(span))