from dataclasses import dataclass, field


@dataclass
class EmissionDO:
    period: str = field()
    agent_ip: str = field()
    eid: str = field()
    energy: float = field()
    flops: float = field()
    co2: float = field()



@dataclass
class ExperimentDO:
    eid: str = field()
    start_time: str = field()
    end_time: str = field()
    status: str = field()
    owner: str = field()


@dataclass
class AgentInfoDO:
    eid: str = field()
    agent_ip: str = field()
    cpu_brand: str = field()
    gpu_brand: str = field()
    cpu_count: int = field()
    gpu_count: int = field()

@dataclass
class TaskInfoDO:
    agent_ip: str = field()
    python_version: str = field()
    os_info: str = field()
    cpu_brand: str = field()
    gpu_brand: str = field()
    cpu_count: int = field()
    gpu_count: int = field()
    username: str = field()


@dataclass
class EmissionChartDO:
    date: str = field()
    energy: float = field()
    flops: float = field()
    co2: flops = field()


@dataclass
class EmissionAggDO:
    energy_total: float = field()
    flops_total: float = field()
    co2_total: float = field()