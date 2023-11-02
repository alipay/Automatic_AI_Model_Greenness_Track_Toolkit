from greenness_track_toolkit.agent.core.colletor.collector_base import Collector
from greenness_track_toolkit.agent.core.colletor.energy.cpu.util.cpu_info import CPUBrand
from greenness_track_toolkit.agent.core.colletor.energy.cpu.intel.intel_rapl_msr import INTEL_RAPL_MSR
from greenness_track_toolkit.agent.core.colletor.energy.cpu.amd.amd_rapl_msr import AMD_RAPL_MSR
from greenness_track_toolkit.agent.core.colletor.energy.cpu.util.cpu_info import CPU
from greenness_track_toolkit.agent.models.energy import Energy
import psutil
import os


def delta_pcputimes(p1, p2):
    p1_total = p1.user + p1.system
    p2_total = p2.user + p2.system
    assert p2_total >= p1_total
    return p2_total - p1_total


def delta_scputimes(s1, s2):
    s1_total = s1.user + s1.system + s1.nice + s1.irq + s1.softirq
    s2_total = s2.user + s2.system + s2.nice + s2.irq + s2.softirq
    assert s2_total >= s1_total
    return s2_total - s1_total


class CPUCollector(Collector):
    def close(self):
        pass

    def start(self):
        self._collector.start()
        self.latest_pcputimes = psutil.Process(self._pid).cpu_times()
        self.latest_scputimes = psutil.cpu_times()

    def delta(self, duration) -> Energy:
        """
        Return the energy consumption for a period of time
        :param duration: default is 0.5s
        :return:
        """
        energy_total = self._collector.delta(duration)
        # the cpu utilization of the process
        pcputimes = psutil.Process(self._pid).cpu_times()
        # the cpu utilization of whole computer
        scputimes = psutil.cpu_times()
        delta_p = delta_pcputimes(self.latest_pcputimes, pcputimes)
        delta_s = delta_scputimes(self.latest_scputimes, scputimes)
        self.latest_pcputimes = pcputimes
        self.latest_scputimes = scputimes
        ratio = delta_p / delta_s
        if ratio > 1.0:
            ratio = 1.0
        return Energy(energy_total * ratio)

    def __init__(self):
        super().__init__()
        self._cpu = CPU()
        self.latest_pcputimes = None
        self.latest_scputimes = None
        # getting pid of the process
        self._pid = os.getpid()
        if self._cpu.brand == CPUBrand.INTEL:
            self._collector = INTEL_RAPL_MSR(self._cpu)
        elif self._cpu.brand == CPUBrand.AMD:
            self._collector = AMD_RAPL_MSR(self._cpu)