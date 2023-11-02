import datetime

from greenness_track_toolkit.agent.config import GLOBAL_CONFIG, TF_CONFIG
from greenness_track_toolkit.agent.models.do import ComputerInfo
from greenness_track_toolkit.rpc.agent_info_service_pb2 import EidRequest, EidResponse, UploadComputerInfoRequest, \
    UploadStatusRequest
from greenness_track_toolkit.agent.service.agent_info_service import AgentInfoService
from greenness_track_toolkit.rpc.common_pb2 import BaseResponse, ResultCode
from greenness_track_toolkit.utils import to_str_time
from greenness_track_toolkit.common_enum import TaskStatus


def upload_task_status(eid, status: TaskStatus):
    request: UploadStatusRequest = UploadStatusRequest(eid=eid, status=status.value)
    agent_info_service = AgentInfoService()
    response: BaseResponse = agent_info_service.uploadStatus(request)
    del agent_info_service
    return response.resultCode == ResultCode.OK


def upload_computer_info(eid, computer_info: ComputerInfo):
    agent_info_service = AgentInfoService()
    request: UploadComputerInfoRequest = UploadComputerInfoRequest(
        eid=eid,
        agent_ip=computer_info.agent_ip,
        cpuBrand=computer_info.cpu_brand,
        cpuKernelNums=computer_info.cpu_kernel_nums,
        cpuClock=computer_info.cpu_clock,
        gpuBrand=computer_info.gpu_brand,
        gpuClock=computer_info.gpu_clock,
        gpuNums=computer_info.gpu_nums,
        rank=TF_CONFIG.task.index
    )
    response: BaseResponse = agent_info_service.uploadComputerInfo(request)
    return response.resultCode == ResultCode.OK


def get_eid_from_server():
    agent_info_service = AgentInfoService()
    eid_request = EidRequest(
        isMaster=TF_CONFIG.task.index == 0,
        masterAgentHostname=TF_CONFIG.cluster.worker[0].hostname,
        ownerName=GLOBAL_CONFIG.username,
        startTime=to_str_time(datetime.datetime.now())
    )
    response: EidResponse = agent_info_service.getUniqueEid(eid_request)
    agent_info_service.__del__()
    return response.eid


def save_task_info(computer_info: ComputerInfo):
    from greenness_track_toolkit.agent.core.dao.task_info_dao import TaskInfoDao
    dao = TaskInfoDao()
    dao.insert(computer_info)