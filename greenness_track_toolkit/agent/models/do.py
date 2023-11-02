from dataclasses import dataclass, field
from greenness_track_toolkit.utils import good_view_time_format, time_diff


@dataclass
class CollectorDO:
    period: str = field()
    energy: float = field()
    flops: float = field()
    co2: float = field()

    def __str__(self):
        return f"{good_view_time_format(self.period)} FLOPs:{self.flops} GFLOPs,Energy:{self.energy} kWh,CO2:{self.co2} kg"


@dataclass
class CollectorSummary:
    start_time: str = field()
    end_time: str = field()
    energy: float = field()
    flops: float = field()
    co2: float = field()

    def __str__(self):
        return f"\n\n" \
               f"-----------------------------------------------------------------------------\n" \
               f"Automatic AI Model Greenness Track Toolkit \n" \
               f"{good_view_time_format(self.start_time)}~{good_view_time_format(self.end_time)} \n" \
               f"time-consuming:\t{time_diff(self.start_time, self.end_time)} seconds\n" \
               f"FLOPs:\t{self.flops} GFLOPs\n" \
               f"Energy:\t{self.energy} kWh\n" \
               f"CO2:\t{self.co2} kg\n" \
               f"-----------------------------------------------------------------------------\n\n"


@dataclass
class ComputerInfo:
    cpu_brand: str = field()
    cpu_kernel_nums: int = field()
    cpu_clock: str = field()
    gpu_brand: str = field()
    gpu_nums: int = field()
    gpu_clock: str = field()
    agent_ip: str = field()
    python_version: str = field()
    os_info: str = field()