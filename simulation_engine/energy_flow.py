class EnergyFlow:
    def __init__(self, load_model, solar_model, battery_model):
        self.load_model = load_model
        self.solar_model = solar_model
        self.battery_model = battery_model

        self._initialize_results()

    def _initialize_results(self):
        self.results = {
            "load": [],
            "solar_available": [],
            "solar_used": [],
            "battery_charge": [],
            "battery_discharge": [],
            "curtailed_solar": [],
            "grid": [],
            "soc": []
        }

    def simulate(self, hours):
        self._initialize_results()

        for h in range(hours):
            load = self.load_model.get_load(h)
            solar = self.solar_model.get_generation(h)
            solar_available = solar

            remaining_load = load

            # Solar → Load
            solar_to_load = min(solar, remaining_load)
            remaining_load -= solar_to_load
            solar -= solar_to_load

            # Solar → Battery
            battery_charge = self.battery_model.charge(solar) if solar > 0 else 0
            solar -= battery_charge

            # Battery → Load
            battery_discharge = 0
            if remaining_load > 0:
                battery_discharge = self.battery_model.discharge(remaining_load)
                remaining_load -= battery_discharge

            # Grid
            grid = max(0, remaining_load)
            curtailed_solar = max(0, solar)

            # Store
            self.results["load"].append(load)
            self.results["solar_available"].append(solar_available)
            self.results["solar_used"].append(solar_to_load)
            self.results["battery_charge"].append(battery_charge)
            self.results["battery_discharge"].append(battery_discharge)
            self.results["curtailed_solar"].append(curtailed_solar)
            self.results["grid"].append(grid)
            self.results["soc"].append(self.battery_model.get_soc_percent())

    def get_results(self):
        return self.results