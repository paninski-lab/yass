import rx
from yass.streaming.core import DataObserver


class IteratedDataObserver(DataObserver):
    """
    Data observer from iterator
    """

    def __init__(self, iterator, maps):
        self.iterator = iterator
        self.obs = rx.Observable.from_iterator(self.iterator)