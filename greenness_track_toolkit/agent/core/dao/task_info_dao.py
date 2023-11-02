import json

from greenness_track_toolkit.agent.models.do import ComputerInfo
from greenness_track_toolkit.agent.utils.dbutils import get_conn
from greenness_track_toolkit.agent.config import GLOBAL_CONFIG
from greenness_track_toolkit.agent.utils.common import get_tf_config


class TaskInfoDao:
    def insert(self, do: ComputerInfo):
        sql_script = """
        insert into `main`.`task_info`(
            `agent_ip`,
            `python_version`,
            `os_info`,
            `cpu_brand`,
            `gpu_brand`,
            `cpu_count`,
            `gpu_count`,
            `username`,
            'tf_config'
        )
        values (
            :agent_ip,
            :python_version,
            :os_info,
            :cpu_brand,
            :gpu_brand,
            :cpu_count,
            :gpu_count,
            :username,
            :tf_config
        )
        """
        parameters = {
            'agent_ip': do.agent_ip,
            'python_version': do.python_version,
            'os_info': do.os_info,
            'cpu_brand': do.cpu_brand,
            'gpu_brand': do.gpu_brand,
            'cpu_count': do.cpu_kernel_nums,
            'gpu_count': do.gpu_nums,
            'username': GLOBAL_CONFIG.username,
            'tf_config': json.dumps(get_tf_config())
        }
        conn = get_conn()
        conn.execute(sql_script, parameters)
        conn.commit()
        conn.close()