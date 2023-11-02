import datetime
import threading
import time

from greenness_track_toolkit.agent.core.colletor.energy.energy_collector import EnergyCollector
from greenness_track_toolkit.agent.core.colletor.flops.flops_collector import FlopsCollector
from greenness_track_toolkit.agent.models.energy import Energy
from greenness_track_toolkit.agent.models.flops import Flops
from greenness_track_toolkit.agent.config import GLOBAL_CONFIG, TF_CONFIG
from greenness_track_toolkit.agent.core.dao.collector import CollectorDao
from greenness_track_toolkit.agent.models.do import CollectorDO, CollectorSummary
from greenness_track_toolkit.utils import get_logger, to_str_time
from greenness_track_toolkit.rpc.emission_service_pb2 import InsertEmissionRequest


class UploadTaskProvider(threading.Thread):
    def __init__(self, task_q):
        super().__init__()
        self._task_q = task_q
        self.end_time = None
        GLOBAL_CONFIG.task_q_signal = True

    def run(self) -> None:
        get_logger().info("date upload working")
        while GLOBAL_CONFIG.working:
            if self.end_time is None:
                self.end_time = datetime.datetime.now()
                start_time = self.end_time - datetime.timedelta(seconds=GLOBAL_CONFIG.upload_server_duration)
            elif (datetime.datetime.now() - self.end_time).seconds >= GLOBAL_CONFIG.upload_server_duration:
                start_time = self.end_time
                self.end_time = self.end_time + datetime.timedelta(seconds=GLOBAL_CONFIG.upload_server_duration)
            else:
                time.sleep(1)
                continue
            self._task_q.append([start_time, self.end_time])
        try:
            GLOBAL_CONFIG.task_q_signal_lock.acquire()
            GLOBAL_CONFIG.task_q_signal = False
            self._task_q.append([self.end_time, datetime.datetime.now()])
        finally:
            GLOBAL_CONFIG.task_q_signal_lock.release()
        get_logger().info("data upload task finish")


class UploadTaskConsumer(threading.Thread):
    def __init__(self, task_q):
        from greenness_track_toolkit.agent.service.emission_service import EmissionService
        super().__init__()
        self._task_q = task_q
        self._dao = CollectorDao()
        self._emission_service = EmissionService()
        get_logger().info("upload consumer")

    def run(self) -> None:
        while GLOBAL_CONFIG.task_q_signal or len(self._task_q) > 0:
            if len(self._task_q) > 0:
                start_time, end_time = self._task_q.pop()
                start_time = to_str_time(start_time)
                end_time = to_str_time(end_time)
                do = self._dao.selectByPeriod(start_time, end_time)
                request: InsertEmissionRequest = InsertEmissionRequest(
                    eid=GLOBAL_CONFIG.eid,
                    agent_ip=GLOBAL_CONFIG.agent_ip,
                    co2=do.co2,
                    energy=do.energy,
                    flops=do.flops,
                    upload_time=end_time
                )
                self._emission_service.insertEmissionData(request=request)
                get_logger().info(f"{start_time}, {end_time} upload, \ncontent:{request}")


class CollectorService:

    def __init__(self, model, session, batch_size, estimator):
        self._tf_config = TF_CONFIG
        self._energy_collector = EnergyCollector()
        self._flops_collector = FlopsCollector(model, session, batch_size, estimator)
        self._dao = CollectorDao()
        if GLOBAL_CONFIG.server_mode:
            self.task_q = []
            self._task_thread = UploadTaskProvider(task_q=self.task_q)
            self._upload_thread = UploadTaskConsumer(task_q=self.task_q)
            self._task_thread.daemon = True
            self._upload_thread.daemon = True

    def start_collector(self):
        self._energy_collector.start()
        self._flops_collector.start()
        if GLOBAL_CONFIG.server_mode:
            self._task_thread.start()
            self._upload_thread.start()
        get_logger().info("start collector")
        pass

    def collect(self):
        energy: Energy = self._energy_collector.delta(GLOBAL_CONFIG.duration)
        flops: Flops = self._flops_collector.delta(GLOBAL_CONFIG.duration)
        do: CollectorDO = CollectorDO(
            period=to_str_time(datetime.datetime.now()),
            energy=energy.convert_to_kWh(),
            flops=flops.convert_gflops(),
            co2=energy.convert_to_co2()
        )
        get_logger().info(do)
        self._dao.insert(do)

    def summary(self):
        do: CollectorSummary = self._dao.summary()
        get_logger().info(do)
        pass

    def __del__(self):
        if GLOBAL_CONFIG.server_mode:
            self._task_thread.join()
            self._upload_thread.join()


class CollectorThread(threading.Thread):
    def __init__(self, collector_service: CollectorService):
        super().__init__()
        self._service = collector_service

    def run(self) -> None:
        self._service.start_collector()
        get_logger().info("collector starting")
        while GLOBAL_CONFIG.working:
            time.sleep(GLOBAL_CONFIG.duration)
            self._service.collect()
        get_logger().info("collector finish")
        self._service.summary()