from greenness_track_toolkit.agent.core.colletor.collector_base import Collector
from greenness_track_toolkit.agent.models.flops import Flops
from greenness_track_toolkit.utils import get_logger


class FlopsCollector(Collector):

    def close(self):
        pass

    def __init__(self, model, session, batch_size: int,
                 estimator=None):
        super().__init__()
        self._logger = get_logger()
        self._flops_collectors = []
        try:
            import tensorflow as tf
            self._logger.info(f"tensorflow version:{tf.__version__}")
            if str(tf.__version__).startswith("1"):
                # init data collector
                if session is None:
                    session = tf.keras.backend.get_session()
                from greenness_track_toolkit.agent.core.colletor.flops.tensorflow1x.tf1_collection import \
                    TF1FlopsCollector
                collector = TF1FlopsCollector(model, session, batch_size, estimator)
                self._flops_collectors.append(collector)
                self._logger.info(f"Building TF1_Collector")

        except Exception as e:
            self._logger.error(f"FLOPs collector init error {e}")

    def start(self):
        [collector.start() for collector in self._flops_collectors]
        pass

    def delta(self, duration) -> Flops:
        flops_total = sum([collector.delta(duration).flops for collector in self._flops_collectors])
        return Flops(flops_total)
        pass