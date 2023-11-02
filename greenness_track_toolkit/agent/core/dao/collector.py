from greenness_track_toolkit.agent.utils.dbutils import get_conn
from greenness_track_toolkit.agent.models.do import CollectorDO,CollectorSummary


class CollectorDao:

    def insert(self, do: CollectorDO):
        sql_script = """
            insert into  `main`.`collect_result`(`period`,`energy`,`flops`,`co2`)
            values (:period,:energy,:flops,:co2)
        """
        parameters = {
            'period': do.period,
            'energy': do.energy,
            'flops': do.flops,
            'co2': do.co2
        }
        conn = get_conn()
        conn.execute(sql_script, parameters)
        conn.commit()
        conn.close()

    def selectByPeriod(self, start_time, end_time) -> CollectorDO:
        sql_script = """
            select max(`period`) as `period`,coalesce( sum(`energy`),0) as `energy`,coalesce(sum(`flops`),0) as `flops`,coalesce(sum(`co2`),0) as `co2`
            from `main`.`collect_result`
            where `period` between :start_time and :end_time
        """
        parameters = {
            "start_time": start_time,
            "end_time": end_time
        }
        conn = get_conn()
        cursor = conn.cursor()
        data = cursor.execute(
            sql_script,
            parameters
        ).fetchone()
        do = CollectorDO(data[0], data[1], data[2], data[3])

        cursor.close()
        conn.close()
        return do

    def summary(self) -> CollectorSummary:
        sql_script = """
            select min(`period`) as `start_time`,max(`period`) as `end_time`,coalesce( sum(`energy`),0) as `energy`,coalesce(sum(`flops`),0) as `flops`,coalesce(sum(`co2`),0) as `co2`
            from `main`.`collect_result`
        """
        conn = get_conn()
        cursor = conn.cursor()
        data = cursor.execute(
            sql_script
        ).fetchone()
        do = CollectorSummary(data[0], data[1], data[2], data[3], data[4])
        cursor.close()
        conn.close()
        return do