from dataclasses import field, dataclass
from grpc import Server


@dataclass
class ServerConfig:
    db_save_path: str = field(init=False)
    server_mode: str = field(init=False)  # Local Server
    rpc_server: Server = field(init=False)
    log_path: str = field(init=False)


GLOBAL_CONFIG = ServerConfig()