from dataclasses import dataclass, field
from typing import List
import json


@dataclass
class Worker:
    hostname: str = field()
    port: int = field(default="")


@dataclass
class Cluster:
    ps: List[Worker] = field()
    worker: List[Worker] = field()


@dataclass
class Task:
    type: str = field()
    index: int = field()


@dataclass
class TFConfig:
    cluster: Cluster = field(init=False)
    task: Task = field(init=False)

    @staticmethod
    def convert_from_json(json_str: str):

        tf_config_json = json.loads(json_str)
        ps = []
        for line in tf_config_json['cluster']['ps']:
            hostname, port = line.split(":")
            worker = Worker(hostname, int(port))
            ps.append(worker)
        workers = []
        for line in tf_config_json['cluster']['worker']:
            hostname, port = line.split(":")
            worker = Worker(hostname, int(port))
            workers.append(worker)

        cluster = Cluster(ps=ps, worker=workers)

        task = Task(tf_config_json['task']['type'], index=tf_config_json['task']['type'])
        tf_config = TFConfig()
        tf_config.cluster = cluster
        tf_config.task = task
        return tf_config