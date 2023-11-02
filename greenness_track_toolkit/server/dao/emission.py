import datetime

from greenness_track_toolkit.server.utils import dbutils
from greenness_track_toolkit.server.models.do import EmissionDO, ExperimentDO, EmissionChartDO, EmissionAggDO
from typing import List

from greenness_track_toolkit.utils import to_str_time


class EmissionDao:
    def insert_emission(self, do: EmissionDO):
        conn = dbutils.get_conn()
        sql = """
        insert into `main`.`collector_result`(`period`,`agent_ip`,`eid`,`energy`,`flops`,`co2`) 
        values(:period,:agent_ip,:eid,:energy,:flops,:co2)
        """
        params = {
            'period': do.period,
            "agent_ip": do.agent_ip,
            "eid": do.eid,
            "energy": do.energy,
            "flops": do.flops,
            "co2": do.co2
        }
        conn.execute(sql, params)
        conn.commit()
        conn.close()

    def select_experiments_list_count(
        self,
        owner='',
        start_time='19000101000000.000000',
        end_time=to_str_time(
            datetime.datetime.now()
        )
    ):
        conn = dbutils.get_conn()
        sql = """
                    SELECT  count(1)
                    FROM    `main`.`experiments`
                    where owner like :owner
                    and start_time between :start_time and :end_time
                """
        cursor = conn.cursor()
        params = {
            'owner': "%" + owner + "%",
            'start_time': start_time,
            'end_time': end_time
        }
        res = cursor.execute(sql, params).fetchone()[0]
        cursor.close()
        conn.close()
        return res

    def select_experiments_list(
        self,
        page_no,
        page_size,
        owner='',
        start_time='19000101000000.000000',
        end_time=to_str_time(
            datetime.datetime.now()
        )
    ) -> List[ExperimentDO]:
        conn = dbutils.get_conn()
        sql = """
            SELECT  eid
                    ,owner
                    ,start_time
                    ,end_time
                    ,status
            FROM    `main`.`experiments`
            where owner like :owner
            and start_time between :start_time and :end_time
            LIMIT ( :page_no * :page_size ) , :page_size
        """
        cursor = conn.cursor()
        params = {
            'owner': "%" + owner + "%",
            'page_no': page_no - 1,
            "page_size": page_size,
            'start_time': start_time,
            'end_time': end_time
        }
        dataset = cursor.execute(sql, params).fetchall()
        res = []
        for row in dataset:
            res.append(ExperimentDO(eid=row[0], owner=row[1], start_time=row[2], end_time=row[3], status=row[4]))
        cursor.close()
        conn.close()
        return res

    def select_experiment_by_eid(self, eid):
        conn = dbutils.get_conn()
        sql = """
            select eid,start_time,end_time,status,owner
            from `main`.`experiments`
            where eid=:eid
                        """
        params = {
            'eid': eid
        }
        cursor = conn.cursor()
        date = cursor.execute(sql, params).fetchone()
        if date is None:
            return None
        cursor.close()
        conn.close()
        return ExperimentDO(eid=date[0], start_time=date[1], end_time=date[2], status=date[3], owner=date[4])

    def select_emission_line_by_eid(
        self,
        eid
    ) -> List[EmissionChartDO]:
        conn = dbutils.get_conn()
        sql = """
            select `period`,sum(energy) energy,sum(flops) flops,sum(co2) co2
            from `main`.`collector_result`
            where eid = :eid
            group by `period`
        """
        params = {
            'eid': eid
        }
        cursor = conn.cursor()
        dataset = cursor.execute(sql, params).fetchall()
        res = []
        for row in dataset:
            res.append(EmissionChartDO(date=row[0], energy=row[1], flops=row[2], co2=row[3]))
        cursor.close()
        conn.close()
        return res

    def select_emission_by_eid(self, eid) -> EmissionAggDO:
        conn = dbutils.get_conn()
        sql = """
                    select sum(energy) energyTotal,sum(flops) flopsTotal,sum(co2) co2Total
                    from `main`.`collector_result`
                    where eid=:eid
                """
        params = {
            'eid': eid
        }
        cursor = conn.cursor()
        date = cursor.execute(sql, params).fetchone()
        cursor.close()
        conn.close()
        return EmissionAggDO(energy_total=date[0], flops_total=date[1], co2_total=date[2])