import threading
from dataclasses import field, dataclass


@dataclass
class CollectorConfig:
    from greenness_track_toolkit.agent.core.service.collector_service import CollectorService
    collector: CollectorService = field(init=False, default=None)
    collector_thread: threading.Thread = field(init=False, default=None)
    global_step: int = field(init=False, default=None)


GLOBAL_COLLECTOR_CONFIG: CollectorConfig = CollectorConfig()