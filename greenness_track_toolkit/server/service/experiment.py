from greenness_track_toolkit.server.dao.emission import EmissionDao, ExperimentDO, EmissionChartDO, EmissionAggDO
from greenness_track_toolkit.server.dao.agent_info import AgentInfoDao, AgentInfoDO
from greenness_track_toolkit.server.models.request import ExperimentListRequest
from greenness_track_toolkit.server.models.response import ExperimentListResponse, ExperimentResponse, DetailExperimentResponse
from typing import List, Optional
from greenness_track_toolkit.utils import good_view_time_format


class ExperimentService:
    def __init__(self):
        self._dao = EmissionDao()
        self._agent_info_dao = AgentInfoDao()

    def select_by_page(self, request: ExperimentListRequest) -> ExperimentListResponse:
        experiment_list: List[ExperimentDO] = self._dao.select_experiments_list(
            page_no=request.pageNum,
            page_size=request.pageSize,
            owner=request.owner,
            start_time=request.startTime,
            end_time=request.endTime
        )
        total: int = self._dao.select_experiments_list_count(
            owner=request.owner,
            start_time=request.startTime,
            end_time=request.endTime
        )
        res = []
        for d in experiment_list:
            agents: List[AgentInfoDO] = self._agent_info_dao.select_agents_by_eid(d.eid)
            cpu_nums = sum([agent.cpu_count for agent in agents])
            gpu_nums = sum([agent.gpu_count for agent in agents])
            exp = ExperimentResponse(
                eid=d.eid,
                owner=d.owner,
                createTime=good_view_time_format(d.start_time),
                endTime=good_view_time_format(d.end_time),
                cpuNun=cpu_nums,
                gpuNum=gpu_nums,
                status=d.status
            )
            res.append(exp)
        return ExperimentListResponse(
            total=total, page_size=request.pageSize, page_no=request.pageNum,
            list=res
        )

    def select_experiment_by_eid(self, eid) -> Optional[DetailExperimentResponse]:
        experiment_info: ExperimentDO = self._dao.select_experiment_by_eid(eid)
        if experiment_info is None:
            return None
        agents: List[AgentInfoDO] = self._agent_info_dao.select_agents_by_eid(eid)
        emission_total: EmissionAggDO = self._dao.select_emission_by_eid(eid)
        emission_line: List[EmissionChartDO] = self._dao.select_emission_line_by_eid(eid)
        return DetailExperimentResponse.convert_from_do(
            agents=agents,
            experiment_info=experiment_info,
            emission_total=emission_total,
            emission_line=emission_line
        )