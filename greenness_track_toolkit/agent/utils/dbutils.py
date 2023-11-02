import sqlite3
from greenness_track_toolkit.agent.config import GLOBAL_CONFIG


def get_conn():
    conn = sqlite3.connect(GLOBAL_CONFIG.db_save_path)
    return conn


def init_agent_table():
    sql = """
        create table if not exists `main`.`collect_result`(
            `period` timestamp primary key,
            `energy` decimal(200,5) not null  default 0.0,
            `flops` decimal(200,5) not null  default 0.0,
            `co2` decimal(200,5) not null  default  0.0
        );
        create table if not exists `main`.`task_info`(
            `agent_ip` varchar(255),
            `python_version` varchar(255),
            `os_info` varchar(255),
            `cpu_brand`  varchar(255) not null  default '',
            `gpu_brand`  varchar(255) not null  default '',
            `cpu_count` int default 0,
            `gpu_count` int default 0,
            `username` varchar(255) not null default '',
            `tf_config` varchar(255) not null default '',
            primary key (`agent_ip`)
        );
    """
    conn = get_conn()
    conn.executescript(sql)
    conn.commit()
    conn.close()