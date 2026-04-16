# application/scenario_generator.py

class ScenarioGenerator:
    """
    Scenario Generator:
    Generates simulation scenarios based on:
    - Budget
    - Resilience requirement
    """

    def __init__(self):
        pass

    def generate(self):
        """
        Returns a list of scenarios.
        Each scenario is a dict that can be fed into HybridSystemSimulator.
        """

        scenarios = [
            {
                "name": "Low Budget",
                "solar_kw": 3,
                "battery_kwh": 5,
                "battery_charge_kw": 2,
                "battery_discharge_kw": 2,
                "priority": "cost"
            },
            {
                "name": "Balanced",
                "solar_kw": 5,
                "battery_kwh": 10,
                "battery_charge_kw": 3,
                "battery_discharge_kw": 3,
                "priority": "balanced"
            },
            {
                "name": "High Resilience",
                "solar_kw": 8,
                "battery_kwh": 20,
                "battery_charge_kw": 5,
                "battery_discharge_kw": 5,
                "priority": "backup"
            }
        ]

        return scenarios


if __name__ == "__main__":
    sg = ScenarioGenerator()
    for sc in sg.generate():
        print(sc)
