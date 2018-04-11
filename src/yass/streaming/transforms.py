from yass.streaming.core import StreamingTransform


class FunctionalStreamingTransform(StreamingTransform):
    """
    Streaming Transform based on a function
    """

    def __init__(self, function):
        self.function = function