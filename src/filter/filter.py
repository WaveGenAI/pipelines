"""
Abstract class for filters.
"""

import abc


class Filter(abc.ABC):
    """
    Abstract class for filters
    """

    @abc.abstractmethod
    def filter(self, filename: str):
        """Abstract method for filtering data.

        Args:
            filename (str): The filename to filter.
        """

        raise NotImplementedError

    @abc.abstractmethod
    def type(self) -> str:
        """Abstract method for getting the type of filter.

        Returns:
            str: The type of filter.
        """

        raise NotImplementedError
