from dataclasses import dataclass, field
from greenness_track_toolkit.agent.config import GLOBAL_CONFIG


def kilowatt_hour_2_joule(kilowatt_hour):
    return kilowatt_hour / (3.6 * 1e6)


@dataclass
class Energy:
    energy: float = field(compare=True)

    def convert_to_kWh(self) -> float:
        # 1 kWh = 3.6 * 10e6 J
        return kilowatt_hour_2_joule(self.energy)

    def convert_to_co2(self) -> float:
        # co2 = kWh * co2 intensity
        # 1 kg = 1000 g

        return self.convert_to_kWh() * GLOBAL_CONFIG.co2_intensity.get_value / 1000
