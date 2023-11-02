import tensorflow as tf

from greenness_track_toolkit import Agent


class AgentSessionHook(tf.train.SessionRunHook):
    def __init__(
        self,
        username: str,
        estimator: tf.estimator.Estimator,
        batch_size: int = 1,
        co2signal_api_key=None,
        log_path="./",
        server: str = None,
        duration=1,
        upload_server_duration=5
    ):
        self.agent = None
        self._username = username
        self._batch_size = batch_size
        self._log_path = log_path
        self._server = server
        self._duration = duration
        self._upload_server_duration = upload_server_duration
        self._estimator = estimator
        self._co2signal_api_key = co2signal_api_key

    def after_create_session(self, session, coord):
        self.agent = Agent(
            username=self._username,
            batch_size=self._batch_size,
            co2signal_api_key=self._co2signal_api_key,
            session=session,
            log_path=self._log_path,
            server=self._server,
            duration=self._duration,
            upload_server_duration=self._upload_server_duration,
            estimator=self._estimator
        )
        self.agent.start()

    def end(self, session):
        self.agent.stop()