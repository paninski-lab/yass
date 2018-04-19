import rx


class StreamingPipeline:
    """
    Pipeline for streaming data. Produce data from a data observer and push
    to data through chain of streaming transforms.
    """

    def __init__(self, data_obs, components):
        self.data_obs = data_obs
        self.components = components
        self._subscribe()

    def _subscribe(self):
        """
        Subscribe pipeline componenets
        """
        pass

    def run(self):
        """
        Start the data observer and flow data through the chain
        """
        self.data_obs.start()


class DataObserver:
    """
    Observer for external data source, this should be overidden
    when plugging in different data sources.
    """

    def __init__(self, obs, maps):
        self.obs = obs
        self.maps = maps

    def start(self):
        """
        Start data observer
        """
        pass


class StreamingTransformation(rx.Observer, rx.Observable):
    """
    Transform for mid pipeline. Subscribes to other piped
    transforms or data observers, applies a transformation
    then pushes data to observers TODO: Buffering
    """

    def __init__(self):
        self._observable = rx.subjects.Subject()

    def subscribe(self,
                  on_next=None,
                  on_error=None,
                  on_completed=None,
                  observer=None):
        """
        Subscribe an observable to the transformer. The Parameters here allow
        the transformer to be subscribed to as a normal rx.observable
        """
        self._observable.subscribe(
            on_next=on_next,
            on_error=on_error,
            on_completed=on_completed,
            observer=observer)

    def process_data(self, x):
        """
        Process data in this transformer, by default this is identity, but
        can be more exotic
        """
        return x

    def on_next(self, x):
        """
        Overriden rx.observer on_next method
        """

        processed = self.process_data(x)
        self._observable.on_next(processed)

    def on_error(self, error):
        """
        Overriden rx.observer on_error method
        """
        self._observable.on_error(error)

    def on_completed(self):
        """
        Overriden rx.observer on_completed method
        """
        self._observable.on_completed()
