from greenness_track_toolkit.agent.core.colletor.collector_base import Collector
from greenness_track_toolkit.agent.core.colletor.energy.gpu.nvidia.nvidia_smi import NvidiaCollector
from greenness_track_toolkit.agent.models.energy import Energy


class GPUCollector(Collector):
    def close(self):
        self._collector.close()

    def start(self):
        self._collector.start()
        pass

    def delta(self, duration) -> Energy:
        return self._collector.delta(duration)
        pass

    def __init__(self):
        super().__init__()
        self._collector = NvidiaCollector()
