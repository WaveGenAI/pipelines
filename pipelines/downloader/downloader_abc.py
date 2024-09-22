"""
Abstract class for downloading data from the internet.
"""

import abc
from typing import List


class Downloader(abc.ABC):
    """
    Downloader class.
    """

    @abc.abstractmethod
    def download_all(self, urls: List[str]) -> None:
        """
        Download the data from the internet.

        Args:
            urls (List[str]): List of URLs to download.
            output_dir (str): The output directory.

        Returns:
            None
        """
        raise NotImplementedError

    @abc.abstractmethod
    def _write_index(self, url, index: int) -> None:
        """
        Write the index to a file.

        Args:
            index (int): The index to write.

        Returns:
            None
        """
        raise NotImplementedError
