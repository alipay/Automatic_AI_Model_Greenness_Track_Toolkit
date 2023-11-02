import datetime

from greenness_track_toolkit.rpc.agent_info_service_pb2_grpc import AgentInfoServiceServicer
from greenness_track_toolkit.rpc.agent_info_service_pb2 import UploadComputerInfoRequest, EidRequest, UploadStatusRequest, EidResponse
from greenness_track_toolkit.rpc.common_pb2 import BaseResponse, ResultCode
from greenness_track_toolkit.server.dao.agent_info import AgentInfoDao
from greenness_track_toolkit.server.dao.experiment import ExperimentDao
from greenness_track_toolkit.utils import to_str_time
from greenness_track_toolkit.common_enum import TaskStatus

class AgentService(AgentInfoServiceServicer):

    def uploadStatus(self, request: UploadStatusRequest, context):
        end_time = None
        if request.status == TaskStatus.FINISH.value or request.status == TaskStatus.CANCEL.value:
            end_time = to_str_time(datetime.datetime.now())
        self.experiment_dao.update_experiment_status(request.eid, request.status, end_time)
        return BaseResponse(resultCode=ResultCode.OK, message="update status success")

    def getUniqueEid(self, request: EidRequest, context):
        if EidRequest.isMaster:
            eid = self.experiment_dao.generate_unique_eid()
            self.experiment_dao.insert_experiment(
                eid=eid,
                owner=request.ownerName, start_time=request.startTime,
                status=TaskStatus.INITIALIZING.value
            )
        else:
            eid = self.experiment_dao.select_eid_by_master_ip(request.masterAgentHostname)
        return EidResponse(resultCode=ResultCode.OK, eid=eid, message="success")

    def uploadComputerInfo(self, request: UploadComputerInfoRequest, context):
        self._dao.insert_agent_info(
            agent_ip=request.agent_ip,
            cpu_count=request.cpuKernelNums,
            cpu_brand=request.cpuBrand,
            gpu_count=request.gpuNums,
            gpu_brand=request.gpuBrand,
            eid=request.eid,
            rank=request.rank
        )

        return BaseResponse(resultCode=ResultCode.OK, message="insert computer info success")

    def __init__(self):
        self._dao = AgentInfoDao()
        self.experiment_dao = ExperimentDao()