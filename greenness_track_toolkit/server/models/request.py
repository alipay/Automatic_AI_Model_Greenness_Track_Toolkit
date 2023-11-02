from dataclasses import dataclass, field

@dataclass
class ExperimentListRequest:
    pageSize: int = field()
    pageNum: int = field()
    startTime: str = field()
    endTime: str = field()
    owner: str = field()