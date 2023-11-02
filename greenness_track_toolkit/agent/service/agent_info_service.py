from greenness_track_toolkit.rpc.agent_info_service_pb2_grpc import AgentInfoServiceStub
from greenness_track_toolkit.agent.config import GLOBAL_CONFIG
import grpc


class AgentInfoService(AgentInfoServiceStub):
    def __init__(self):
        channel = grpc.insecure_channel(f"{GLOBAL_CONFIG.server_hostname}:{GLOBAL_CONFIG.server_port}")
        super().__init__(channel)
        self.channel = channel

    def __del__(self):
        self.channel.close()
