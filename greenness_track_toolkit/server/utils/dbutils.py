import sqlite3
from greenness_track_toolkit.server.config import GLOBAL_CONFIG


def get_conn(db_file=GLOBAL_CONFIG.db_save_path):
    conn = sqlite3.connect(db_file)
    return conn


def init_server_table():
    sql = """
        create table if not exists `main`.`agent_info`(
            `eid` varchar(255),
            `agent_ip` varchar(255),
            `cpu_brand`  varchar(255) not null  default '',
            `gpu_brand`  varchar(255) not null  default '',
            `cpu_count` int default 0,
            `gpu_count` int default 0,
            `rank` int default 0,
            primary key (`eid`,`agent_ip`,`rank`)
        );
        create table if not exists `main`.`experiments`(
            `eid` varchar(255) primary key ,
            `start_time` varchar(255) not null ,
            `end_time` varchar(255)  ,
            `status` varchar(255) not null ,
            `owner` varchar(255) not null 
        );
        create table if not exists `main`.`collector_result`(
            `period` varchar(255),
            `agent_ip` varchar(255) not null ,
            `eid` varchar(255) not null ,
            `energy` decimal(200,5) not null  default 0.0,
            `flops` decimal(200,5) not null  default 0.0,
            `co2` decimal(200,5) not null  default  0.0,
            primary key (`period`,`agent_ip`,`eid`)
        ) ;
    """
    conn = get_conn()
    conn.executescript(sql)
    conn.commit()
    conn.close()