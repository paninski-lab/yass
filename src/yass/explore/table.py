"""
Create tables with style
"""


class Table(object):
    """

    Notes
    -----
    http://ipython.readthedocs.org/en/stable/config/integrating.html
    """
    def __init__(self, content, header):
        try:
            self._tabulate = __import__('tabulate').tabulate
        except Exception:
            raise ImportError('tabulate is required to use the table module')

        self.content = content
        self.header = header

    @property
    def html(self):
        return self._tabulate(self.content, headers=self.header,
                              tablefmt='html')

    def __str__(self):
        return self._tabulate(self.content, headers=self.header,
                              tablefmt='grid')

    def _repr_html_(self):
        """Integration with Jupyter
        """
        return self.html
