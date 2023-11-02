from greenness_track_toolkit.server.utils.dbutils import get_conn
from typing import List
from greenness_track_toolkit.server.models.do import AgentInfoDO


class AgentInfoDao:
    def insert_agent_info(self, eid, agent_ip, cpu_brand, gpu_brand, cpu_count, gpu_count, rank):
        conn = get_conn()
        sql = """
            insert into `main`.`agent_info`(`eid`,`agent_ip`,`cpu_brand`,`gpu_brand`,`cpu_count`,`gpu_count`,rank) 
            values (:eid,:agent_ip,:cpu_brand,:gpu_brand,:cpu_count,:gpu_count,:rank)
        """
        params = {
            'eid': eid,
            'agent_ip': agent_ip,
            'cpu_brand': cpu_brand,
            'gpu_brand': gpu_brand,
            'cpu_count': cpu_count,
            'gpu_count': gpu_count,
            'rank': rank
        }
        conn.execute(sql, params)
        conn.commit()
        conn.close()
        pass

    def select_agents_by_eid(self, eid) -> List[AgentInfoDO]:
        conn = get_conn()
        sql = """
            select `eid`,`agent_ip`,`cpu_brand`,`gpu_brand`,`cpu_count`,`gpu_count` 
            from `main`.`agent_info`
            where eid=:eid
        """
        params = {
            'eid': eid
        }
        cursor = conn.cursor()
        dataset = cursor.execute(sql, params).fetchall()
        res = []
        for row in dataset:
            res.append(
                AgentInfoDO(
                    eid=row[0],
                    agent_ip=row[1],
                    cpu_brand=row[2],
                    gpu_brand=row[3],
                    cpu_count=row[4],
                    gpu_count=row[5]
                )
            )
        cursor.close()
        conn.close()
        return res
        pass