from greenness_track_toolkit import Agent
import time
a = 1
with Agent(
            # server="localhost:16886",
            username="tangchengfu.tcf",
            session=None,
            batch_size=None,
            log_path="../logs",
) as agent:
        for epoch in range(0, 20):
                a = a + a
                time.sleep(5)