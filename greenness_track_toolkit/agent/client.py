import datetime
import os
import signal
import socket
import sys
import time
import uuid

from greenness_track_toolkit.agent.config import GLOBAL_CONFIG, TF_CONFIG
from greenness_track_toolkit.common_enum import TASK_LOCAL_TYPE, TaskStatus, TF_CONFIG_KEY, \
    TF_MASTER_INDEX
from greenness_track_toolkit.agent.core.process import GLOBAL_COLLECTOR_CONFIG
from greenness_track_toolkit.agent.core.service.agent_service import get_eid_from_server, save_task_info, \
    upload_computer_info, upload_task_status
from greenness_track_toolkit.agent.core.service.collector_service import CollectorService, CollectorThread
from greenness_track_toolkit.agent.models.co2 import CO2Intensity
from greenness_track_toolkit.agent.models.tf_config import Cluster, Task, Worker
from greenness_track_toolkit.agent.utils.common import get_computer_info
from greenness_track_toolkit.agent.utils.dbutils import init_agent_table
from greenness_track_toolkit.utils import get_logger, to_str_time


def _init_config(
    username,
    server,
    log_path,
    model,
    session,
    batch_size,
    duration,
    upload_server_duration,
    estimator,
    co2signal_api_key
):
    if co2signal_api_key:
        GLOBAL_CONFIG.co2_intensity = CO2Intensity(co2signal_api_key=co2signal_api_key)
    GLOBAL_CONFIG.status = TaskStatus.INITIALIZING
    GLOBAL_CONFIG.start_time = to_str_time(datetime.datetime.now())
    hostname = socket.gethostname()
    # processing log path
    log_path = os.path.join(log_path, f"{username}_{time.time()}_{uuid.uuid4()}_{hostname}", f"db")
    os.makedirs(os.path.dirname(log_path))
    GLOBAL_CONFIG.db_save_path = log_path
    GLOBAL_CONFIG.username = username
    # init database
    init_agent_table()
    computer_info = get_computer_info()
    save_task_info(computer_info)
    # server mode check
    server_mode = server is not None
    GLOBAL_CONFIG.server_mode = server_mode
    if server_mode and model is None and session is None:
        get_logger().error("server model only support in tensorflow backbone. please set model or session")
        return

    if server_mode and server.find(":") > 0 and len(server.split(":")) == 2:
        server_hostname, server_port = server.split(":")
        # parsing tf config
        tf_config = os.environ.get(TF_CONFIG_KEY, None)
        if tf_config is not None and tf_config != "":
            TF_CONFIG.convert_from_json(tf_config)
        else:
            TF_CONFIG.task = Task(index=0, type=TASK_LOCAL_TYPE)
            TF_CONFIG.cluster = Cluster(ps=[], worker=[Worker(hostname=socket.gethostname())])
        GLOBAL_CONFIG.agent_ip = hostname
        GLOBAL_CONFIG.server_hostname = server_hostname
        GLOBAL_CONFIG.server_port = server_port
        # master node upload task info
        eid = get_eid_from_server()
        GLOBAL_CONFIG.eid = eid
        # upload computer info to server

        upload_computer_info(eid, computer_info)

    GLOBAL_COLLECTOR_CONFIG.collector = CollectorService(model, session, batch_size, estimator)
    # startup energy and flops monitor
    try:
        GLOBAL_CONFIG.working_lock.acquire()
        GLOBAL_CONFIG.working = True
    finally:
        GLOBAL_CONFIG.working_lock.release()
    GLOBAL_CONFIG.duration = duration
    GLOBAL_CONFIG.upload_server_duration = upload_server_duration
    GLOBAL_COLLECTOR_CONFIG.collector_thread = CollectorThread(GLOBAL_COLLECTOR_CONFIG.collector)


def _is_master():
    return GLOBAL_CONFIG.server_mode and TF_CONFIG.task.index == TF_MASTER_INDEX


class Agent:

    def __init__(
        self,
        username: str,
        batch_size: int = 1,
        co2signal_api_key: str = None,
        model=None,
        session=None,
        log_path="./",
        server: str = None,
        duration=1,
        upload_server_duration=5,
        estimator=None
    ):
        assert duration < upload_server_duration, get_logger().error("the parameter duration must greater than "
                                                                     "upload_server_duration")
        # init global config
        _init_config(
            username, server, log_path, model, session, batch_size, duration, upload_server_duration, estimator,
            co2signal_api_key
        )

    def start(self):
        self._init_signal_processor()
        GLOBAL_COLLECTOR_CONFIG.collector_thread.start()
        GLOBAL_CONFIG.status = TaskStatus.RUNNING
        if _is_master():
            upload_task_status(GLOBAL_CONFIG.eid, GLOBAL_CONFIG.status)

    def _init_signal_processor(self):
        def signal_handler(n, _):
            self.stop(TaskStatus.CANCEL)
            sys.exit(n)

        signal.signal(signal.SIGINT, handler=signal_handler)
        signal.signal(signal.SIGTERM, handler=signal_handler)

    def stop(self, status=TaskStatus.FINISH):
        try:
            GLOBAL_CONFIG.working_lock.acquire()
            GLOBAL_CONFIG.working = False
        finally:
            GLOBAL_CONFIG.working_lock.release()
        GLOBAL_COLLECTOR_CONFIG.collector_thread.join()
        if _is_master():
            upload_task_status(GLOBAL_CONFIG.eid, status)
        get_logger().info("agent closed")

    def __enter__(self):
        self.start()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()