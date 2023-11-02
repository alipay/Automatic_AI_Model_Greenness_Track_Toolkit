from greenness_track_toolkit.rpc import emission_service_pb2
from greenness_track_toolkit.rpc import emission_service_pb2_grpc
from greenness_track_toolkit.rpc import common_pb2
from greenness_track_toolkit.utils import get_logger
from greenness_track_toolkit.server.dao.emission import EmissionDao, EmissionDO


class EmissionService(emission_service_pb2_grpc.EmissionServiceServicer):
    def __init__(self):
        self._dao = EmissionDao()

    def insertEmissionData(self, request: emission_service_pb2.InsertEmissionRequest, context):
        do = EmissionDO(period=request.upload_time, agent_ip=request.agent_ip, eid=request.eid, energy=request.energy,
                        flops=request.flops, co2=request.co2)
        self._dao.insert_emission(do)
        get_logger().info(request)
        return common_pb2.BaseResponse(resultCode=common_pb2.OK, message="insert success")