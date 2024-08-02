"""
Abstract class for downloading data from the internet.
"""

import abc


class Downloader(abc.ABC):
    """
    Downloader class.
    """

    @abc.abstractmethod
    def download(self, url: str, output_dir: str) -> str:
        """
        Download the data from the internet.

        Args:
            url (str): The url to download.
            output_dir (str): The output directory.

        Returns:
            str: The path to the downloaded file.
        """
        raise NotImplementedError
