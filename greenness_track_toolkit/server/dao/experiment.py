import time
import uuid

from greenness_track_toolkit.server.utils import dbutils


class ExperimentDao:
    def insert_experiment(self, eid, owner, start_time, status):
        conn = dbutils.get_conn()
        sql = """
        insert into `main`.`experiments`(`eid`,`start_time`,`status`,`owner`) 
        values (:eid,:start_time,:status,:owner)
        """
        params = {
            'eid': eid,
            'start_time': start_time,
            'status': status,
            'owner': owner
        }
        conn.execute(sql, params)
        conn.commit()
        conn.close()

    def update_experiment_status(self, eid, status, end_time=None):
        conn = dbutils.get_conn()
        sql = f"""
            update  `main`.`experiments` 
            set `status`=:status {",end_time=:end_time" if end_time is not None else ""} 
            where `eid`=:eid
        """
        params = {
            'eid': eid,
            'status': status,
            'end_time': end_time
        }
        conn.execute(sql, params)
        conn.commit()
        conn.close()

    def select_eid_by_master_ip(self, master_ip):
        conn = dbutils.get_conn()
        sql = f"""
                  select eid from `main`.`agent_info`
                  where agent_ip=:master_ip
                  order by eid desc
                  limit 1
                """
        params = {
            'master_ip': master_ip,
        }
        res = conn.execute(sql, params).fetchone()[0]
        conn.close()
        return res

    def generate_unique_eid(self):
        uid = uuid.uuid4()
        current_time = time.time()
        return f"{current_time}{uid}"