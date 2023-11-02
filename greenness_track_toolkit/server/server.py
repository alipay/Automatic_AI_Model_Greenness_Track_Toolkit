import threading

from greenness_track_toolkit.server.utils.exception import DoNotSupportServerMode
from greenness_track_toolkit.utils import get_logger
from greenness_track_toolkit.server.config import GLOBAL_CONFIG


def _init_grpc(_rpc_port):
    import grpc
    from concurrent import futures
    from greenness_track_toolkit.rpc import agent_info_service_pb2_grpc, emission_service_pb2_grpc
    from greenness_track_toolkit.server.service.emission import EmissionService
    from greenness_track_toolkit.server.service.agent import AgentService
    _grpc_server: grpc.Server = grpc.server(futures.ThreadPoolExecutor(max_workers=50))
    agent_info_service_pb2_grpc.add_AgentInfoServiceServicer_to_server(AgentService(), _grpc_server)
    emission_service_pb2_grpc.add_EmissionServiceServicer_to_server(EmissionService(), _grpc_server)
    _grpc_server.add_insecure_port(f"[::]:{_rpc_port}")
    _grpc_server.start()
    GLOBAL_CONFIG.rpc_server = _grpc_server
    _grpc_server.wait_for_termination()


class Server:
    def __init__(self, db_save_path="./db", server_mode="SERVER", rpc_port=16886, api_port=8000, log_path="./"):
        self.logger = get_logger()
        GLOBAL_CONFIG.server_mode = server_mode
        GLOBAL_CONFIG.db_save_path = db_save_path
        self._api_port = api_port
        if GLOBAL_CONFIG.server_mode == "SERVER":
            self._rpc_port = rpc_port
            self._init_server_mode()
        elif GLOBAL_CONFIG.server_mode == "LOCAL":
            GLOBAL_CONFIG.log_path = log_path
        else:
            raise DoNotSupportServerMode()

    def _init_server_mode(self):
        from greenness_track_toolkit.server.utils.dbutils import init_server_table
        init_server_table()
        self._grpc_thread = threading.Thread(target=_init_grpc, args=(self._rpc_port,))

    def start(self):
        if GLOBAL_CONFIG.server_mode == "SERVER":
            self._grpc_thread.start()
            self._start_server_api()
        elif GLOBAL_CONFIG.server_mode == "LOCAL":
            self._start_local_api()
        else:
            raise DoNotSupportServerMode()

    def _start_server_api(self):
        from greenness_track_toolkit.server.api.server.server import app
        app.run(host="0.0.0.0", port=self._api_port)
        pass

    def _start_local_api(self):
        from greenness_track_toolkit.server.api.local.local import app
        app.run(host="0.0.0.0", port=self._api_port)
        pass

    def __del__(self):
        if GLOBAL_CONFIG.server_mode == "SERVER":
            GLOBAL_CONFIG.rpc_server.stop(grace=None)
            self._grpc_thread.join()