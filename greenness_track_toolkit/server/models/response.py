from dataclasses import dataclass, field
from typing import List

from greenness_track_toolkit.server.models.do import EmissionAggDO, EmissionChartDO

from greenness_track_toolkit.utils import good_view_time_format, time_diff


@dataclass
class ExperimentResponse:
    eid: str = field()
    owner: str = field()
    createTime: str = field()
    endTime: str = field()
    cpuNun: int = field()
    gpuNum: int = field()
    status: str = field()
    idx: int = field(default=0)


@dataclass
class ExperimentListResponse:
    total: int = field()
    page_size: int = field()
    page_no: int = field()
    list: List[ExperimentResponse] = field()


@dataclass
class EmissionLineResponse:
    date: str = field()
    value: float = field()


@dataclass
class DetailExperimentResponse:
    eid: str = field()
    cpuType: str = field()
    cpuNum: int = field()
    gpuType: str = field()
    gpuNum: int = field()
    owner: str = field()
    createTime: str = field()
    endTime: str = field()
    energyTotal: float = field()
    flopsTotal: float = field()
    co2Total: float = field()
    energyLine: List[EmissionLineResponse] = field()
    flopsLine: List[EmissionLineResponse] = field()
    co2Line: List[EmissionLineResponse] = field()
    timeSpend: int = field()

    @staticmethod
    def convert_from_do(agents, experiment_info, emission_total, emission_line):
        cpu_types = []
        gpu_types = []
        cpu_num = 0
        gpu_num = 0
        for agent in agents:
            if agent.cpu_brand != '':
                cpu_types.append(agent.cpu_brand)
            if agent.gpu_brand != '':
                gpu_types.append(agent.gpu_brand)
            cpu_num += agent.cpu_count
            gpu_num += agent.gpu_count

        energyLine = []
        co2Line = []
        flopsLine = []
        for line in emission_line:
            energyLine.append(EmissionLineResponse(date=good_view_time_format(line.date), value=line.energy))
            co2Line.append(EmissionLineResponse(date=good_view_time_format(line.date), value=line.co2))
            flopsLine.append(EmissionLineResponse(date=good_view_time_format(line.date), value=line.flops))
        return DetailExperimentResponse(
            eid=experiment_info.eid,
            owner=experiment_info.owner,
            createTime=good_view_time_format(experiment_info.start_time),
            endTime=good_view_time_format(experiment_info.end_time),
            energyTotal=round(emission_total.energy_total, 6),
            flopsTotal=round(emission_total.flops_total, 6),
            co2Total=round(emission_total.co2_total, 6),
            cpuType=str.join(',', cpu_types) if cpu_types is not None and len(cpu_types) != 0 else "-",
            gpuType=str.join(',', gpu_types) if gpu_types is not None and len(gpu_types) != 0 else "-",
            cpuNum=cpu_num,
            gpuNum=gpu_num,
            energyLine=energyLine,
            co2Line=co2Line,
            flopsLine=flopsLine,
            timeSpend=time_diff(experiment_info.start_time, experiment_info.end_time)
        )

    @staticmethod
    def convert_from_local_do(eid, agent, agg: EmissionAggDO, char: List[EmissionChartDO]):
        energy_line = []
        flops_line = []
        co2_line = []
        for line in char:
            energy_line.append(EmissionLineResponse(date=good_view_time_format(str(line.date)), value=line.energy))
            flops_line.append(EmissionLineResponse(date=good_view_time_format(str(line.date)), value=line.flops))
            co2_line.append(EmissionLineResponse(date=good_view_time_format(str(line.date)), value=line.co2))
        return DetailExperimentResponse(
            eid=eid, cpuType=agent.cpu_brand,
            cpuNum=agent.cpu_count, gpuType=agent.gpu_brand, gpuNum=agent.gpu_count
            , owner=agent.username, createTime=good_view_time_format(char[0].date),
            endTime=good_view_time_format(char[-1].date), energyTotal=agg.energy_total,
            flopsTotal=agg.flops_total, co2Total=agg.co2_total,
            energyLine=energy_line, flopsLine=flops_line
            , co2Line=co2_line, timeSpend=time_diff(char[0].date, char[-1].date)
        )
        pass