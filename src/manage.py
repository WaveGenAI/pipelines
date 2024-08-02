""" 
Manager module.
"""

import logging

from src.downloader import DownloaderUrl
from src.filter import DeduplicateFilter
from src.generator import Process

logging.basicConfig(level=logging.INFO)


class Manager:
    """
    Manager class.
    """

    def __init__(self, input_file: str = "data.txt", output_dir: str = "audio"):
        """Constructor for the Manager class.

        Args:
            input_file (str, optional): the input file that contain url. Defaults to "data.txt".
            output_dir (str, optional): the output directory. Defaults to "audio".
        """

        self.filters = [DeduplicateFilter()]
        self._input_file = input_file
        self._output_dir = output_dir

        self._process = Process()

    def run(self) -> None:
        """
        Run the manager.
        """

        for filter_data in self.filters:
            if filter_data.type() == "text":
                filter_data.filter(self._input_file)

        for url in self._process.iter_over_file(self._input_file):
            downloader = DownloaderUrl()
            audio_file = downloader.download(url, self._output_dir)
            logging.info("Downloaded %s", url)

            break
