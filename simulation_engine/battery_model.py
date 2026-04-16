class BatteryModel:
    def __init__(
        self,
        capacity_kwh,
        max_charge_kw,
        max_discharge_kw,
        efficiency=0.9,
        initial_soc=0.5,
        min_soc=0.2,
        max_soc=1.0
    ):
        self.capacity_kwh = capacity_kwh
        self.max_charge_kw = max_charge_kw
        self.max_discharge_kw = max_discharge_kw
        self.efficiency = max(0.0, min(1.0, efficiency))

        self.min_soc_fraction = max(0.0, min(1.0, min_soc))
        self.max_soc_fraction = max(self.min_soc_fraction, min(1.0, max_soc))

        self.soc = max(self.min_soc_fraction, min(self.max_soc_fraction, initial_soc)) * capacity_kwh
        self.min_soc = self.min_soc_fraction * capacity_kwh
        self.max_soc = self.max_soc_fraction * capacity_kwh

        self.total_charged_kwh = 0.0
        self.total_discharged_kwh = 0.0

    def charge(self, energy):
        if energy <= 0:
            return 0.0

        energy = min(energy, self.max_charge_kw)
        space = self.max_soc - self.soc
        charged = min(energy * self.efficiency, max(0.0, space))
        self.soc += charged
        self.total_charged_kwh += charged
        return charged

    def discharge(self, demand):
        if demand <= 0:
            return 0.0

        demand = min(demand, self.max_discharge_kw)
        available = self.soc - self.min_soc
        if self.efficiency == 0:
            return 0.0

        discharged = min(demand / self.efficiency, max(0.0, available))
        self.soc -= discharged
        delivered = discharged * self.efficiency
        self.total_discharged_kwh += delivered
        return delivered

    def get_soc(self):
        return self.soc

    def get_soc_percent(self):
        return (self.soc / self.capacity_kwh) * 100