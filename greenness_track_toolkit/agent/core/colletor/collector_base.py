import abc


class Collector(abc.ABC):
    def __init__(self):
        pass

    @abc.abstractmethod
    def start(self):
        raise NotImplementedError

    @abc.abstractmethod
    def delta(self, duration):
        raise NotImplementedError

    @abc.abstractmethod
    def close(self):
        raise NotImplementedError
