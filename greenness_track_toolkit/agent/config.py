import threading
from dataclasses import dataclass, field

from greenness_track_toolkit.agent.models.co2 import CO2Intensity
from greenness_track_toolkit.agent.models.tf_config import TFConfig


@dataclass
class AgentConfig:
    agent_ip: str = field(init=False)
    server_mode: bool = field(init=False)
    server_hostname: str = field(init=False)
    server_port: int = field(init=False)
    db_save_path: str = field(init=False)
    username: str = field(init=False)
    start_time: float = field(init=False)
    eid: str = field(init=False)
    working: str = field(init=False)
    working_lock: threading.Lock = field(init=False, default=threading.Lock())
    status: str = field(init=False)
    duration: int = field(init=False)
    upload_server_duration: int = field(init=False)
    task_q_signal: bool = field(init=True, default=False)
    task_q_signal_lock: threading.Lock = field(init=False, default=threading.Lock())
    global_step: object = field(init=False, default=None)
    exit_status: bool = field(init=False, default=False)
    co2_intensity: CO2Intensity = field(init=True, default=CO2Intensity())


GLOBAL_CONFIG: AgentConfig = AgentConfig()
TF_CONFIG: TFConfig = TFConfig()