""" 
Manager module.
"""

import logging

from tagging.tagger import Tagger

import src.utils
from src.downloader import DownloaderUrl
from src.exceptions import DownloadUrlException
from src.filter import DeduplicateFilter
from src.generator import LLM

logging.basicConfig(level=logging.INFO)


class Manager:
    """
    Manager class.
    """

    def __init__(
        self,
        input_file: str = "data.txt",
        output_dir: str = "audio",
        batch_size: int = 2,
    ):
        """Constructor for the Manager class.

        Args:
            input_file (str, optional): the input file that contain url. Defaults to "data.txt".
            output_dir (str, optional): the output directory. Defaults to "audio".
            batch_size (int, optional): the batch size. Defaults to 2.
        """

        self.filters = [DeduplicateFilter()]
        self._input_file = input_file
        self._output_dir = output_dir

        self._tagger = Tagger()
        self._tagger.load_model()

        self._llm = LLM()
        self._batch_size = batch_size

        self._downloader = DownloaderUrl()

    def run(self) -> None:
        """
        Run the manager.
        """

        for filter_data in self.filters:
            if filter_data.type() == "text":
                filter_data.filter(self._input_file)

        with open(self._input_file, "r", encoding="utf-8") as f:

            urls = []
            for url in f:
                urls.append(url)

                if len(urls) < self._batch_size:
                    continue

                downloaded_files = []
                for url in urls:
                    print(url)
                    try:
                        audio_file = self._downloader.download(
                            url.strip(), self._output_dir
                        )
                        downloaded_files.append(audio_file)
                    except DownloadUrlException as e:
                        logging.error(e)

                urls = []

                logging.info("Tagging %s", self._batch_size)
                tags = self._tagger.tag(downloaded_files)

                logging.info("Generating descriptions")
                for path, description in tags.items():
                    name = path.split("/")[-1].replace(".mp3", "")
                    out = self._llm.generate(name, description)

                    src.utils.save_prompt(f"{self._output_dir}/{name}.txt", out)
