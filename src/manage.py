""" 
Manager module.
"""

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

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
        batch_size: int = 6,
    ):
        """Constructor for the Manager class.

        Args:
            input_file (str, optional): the input file that contain url. Defaults to "data.txt".
            output_dir (str, optional): the output directory. Defaults to "audio".
            batch_size (int, optional): the batch size. Defaults to 6.
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
            data = []
            for line in f:
                data.append(line.split(";"))

                if len(data) < self._batch_size:
                    continue

                logging.info("Downloading %s", self._batch_size)
                downloaded_files = []

                with ThreadPoolExecutor(max_workers=2) as executor:
                    future_to_url = {
                        executor.submit(
                            self._downloader.download, line[0].strip(), self._output_dir
                        ): line
                        for line in data
                    }
                    for idx, future in enumerate(as_completed(future_to_url)):
                        try:
                            audio_file = future.result()
                            if audio_file:
                                downloaded_files.append(
                                    (audio_file, data[idx][1].strip())
                                )
                        except DownloadUrlException as e:
                            logging.error("Error while downloading %s", e)

                data = []

                if len(downloaded_files) == 0:
                    continue

                logging.info("Tagging %s", len(downloaded_files))
                tags = self._tagger.tag([file for file, _ in downloaded_files])

                logging.info("Generating descriptions")

                lst_tags = [
                    [
                        values[0].split("/")[-1].replace(".mp3", ""),
                        values[1],
                        downloaded_files[idx][1],
                    ]
                    for idx, values in enumerate(tags.items())
                ]

                outputs = self._llm.generate(lst_tags)

                for out in outputs:
                    src.utils.save_prompt(
                        f"{self._output_dir}/{out['name']}.txt", out["description"]
                    )
