import sys

from greenness_track_toolkit.agent.core.colletor.collector_base import Collector
from greenness_track_toolkit.agent.core.colletor.energy.cpu.cpu_collector import CPUCollector
from greenness_track_toolkit.agent.core.colletor.energy.gpu.gpu_collector import GPUCollector
from greenness_track_toolkit.agent.models.energy import Energy
from greenness_track_toolkit.agent.utils.common import get_gpu_visible
from greenness_track_toolkit.utils import get_logger


class EnergyCollector(Collector):

    def __init__(self):
        super().__init__()
        self._energy_collectors = []
        if sys.platform.lower() == "linux":
            try:
                cpu_collector = CPUCollector()
                self._energy_collectors.append(cpu_collector)
            except Exception as e:
                get_logger().warning(f"CPU energy collector init error {e}")
        else:
            get_logger().warning("CPU energy collector only support linux platform")
        if get_gpu_visible():
            try:
                gpu_collector = GPUCollector()
                self._energy_collectors.append(gpu_collector)
            except Exception as e:
                get_logger().warning(f"GPU energy collector init error {e}")

    def start(self):
        [collector.start() for collector in self._energy_collectors]
        pass

    def delta(self, duration) -> Energy:
        energy_total = sum([collector.delta(duration).energy for collector in self._energy_collectors])
        return Energy(energy_total)

    def close(self):
        [collector.close() for collector in self._energy_collectors]