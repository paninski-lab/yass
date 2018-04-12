from yass.streaming.core import StreamingTransformation


class FunctionalStreamingTransformation(StreamingTransformation):
    """
    Streaming Transform based on a function
    """

    def __init__(self, function):
        self.function = function
