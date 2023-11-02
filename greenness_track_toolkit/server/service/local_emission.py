import os

from greenness_track_toolkit.server.models.do import EmissionChartDO, EmissionAggDO, TaskInfoDO
from greenness_track_toolkit.server.models.response import DetailExperimentResponse, ExperimentListResponse, \
    ExperimentResponse
from greenness_track_toolkit.server.utils.dbutils import get_conn
from greenness_track_toolkit.server.config import GLOBAL_CONFIG


class LocalEmissionService:
    def __init__(self):
        self.log_db = []
        for root, _, file in os.walk(top=GLOBAL_CONFIG.log_path, topdown=False):
            if len(file) >= 1 and 'db' in file:
                self.log_db.append(os.path.dirname(os.path.abspath(os.path.join(root, file[0]))))

        print(self.log_db)

    def select_all_db_file(self, pageSize, pageNum):
        list = []
        for idx, log in enumerate(self.log_db[(pageNum - 1) * pageSize:pageNum * pageSize]):
            agent = self.select_agent_info(idx)
            emission = self.select_emission_by_filename(idx)
            start_time, end_time = emission.createTime, emission.endTime
            list.append(
                ExperimentResponse(
                    eid=log,
                    owner=agent.username,
                    createTime=start_time,
                    cpuNun=agent.cpu_count,
                    gpuNum=agent.gpu_count,
                    status="FINISH",
                    endTime=end_time,
                    idx=idx
                )
            )
        return ExperimentListResponse(
            total=len(self.log_db), page_size=pageSize, page_no=pageNum,
            list=list
        )

    def select_agent_info(self, idx) -> TaskInfoDO:
        file_path = self.log_db[idx]
        conn = get_conn(os.path.join(file_path, "db"))

        sql = """
              select `agent_ip`,`python_version`,`os_info`,`cpu_brand`,`gpu_brand`,`cpu_count`,`gpu_count`,`username` 
              from `main`.`task_info`
                """
        cursor = conn.cursor()
        row = cursor.execute(sql).fetchone()
        agent = TaskInfoDO(
            agent_ip=row[0], python_version=row[1], os_info=row[2], cpu_brand=row[3], gpu_brand=row[4],
            cpu_count=row[5], gpu_count=row[6],
            username=row[7])
        cursor.close()
        return agent

    def select_emission_by_filename(self, idx):
        file_path = self.log_db[idx]
        conn = get_conn(os.path.join(file_path, "db"))

        sql = """
        select `period`,`energy`,`flops`,`co2`
        from `main`.`collect_result`
        """
        cursor = conn.cursor()
        res = []
        rows = cursor.execute(sql).fetchall()
        for row in rows:
            res.append(EmissionChartDO(date=row[0], energy=row[1], flops=row[2], co2=row[3]))
        cursor.close()

        sql = """
            select sum(`energy`) energy,sum(`flops`) flops,sum(`co2`) co2
            from `main`.`collect_result`
        """

        cursor = conn.cursor()
        row = cursor.execute(sql).fetchone()
        agg = EmissionAggDO(energy_total=row[0], flops_total=row[1], co2_total=row[2])
        cursor.close()
        agent_info: TaskInfoDO = self.select_agent_info(idx)
        return DetailExperimentResponse.convert_from_local_do(
            file_path.split("/")[-1],
            agent=agent_info,
            agg=agg,
            char=res
        )