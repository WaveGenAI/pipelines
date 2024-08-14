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
    def download_all(self, urls: List[str], output_dir: str) -> None:
        """
        Download the data from the internet.

        Args:
            urls (List[str]): List of URLs to download.
            output_dir (str): The output directory.

        Returns:
            None
        """
        raise NotImplementedError
