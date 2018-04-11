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


class StreamingTransform(rx.Subject):
    """
    Transform for mid pipeline. Subscribes to other piped
    transforms or data observers, applies a transformation
    then pushes data to observers
    """

    def __init__(self):
        pass
