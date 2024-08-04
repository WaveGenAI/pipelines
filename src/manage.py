""" 
Manager module.
"""

import asyncio
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import soundfile
from tagging.tagger import Tagger

import src.utils
from src.downloader import DownloaderUrl
from src.exceptions import DownloadUrlException
from src.filter import DeduplicateFilter, SpaceFilter
from src.generator import LLM, FeatureExtractor

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

        self.filters = [DeduplicateFilter(), SpaceFilter()]
        self._input_file = input_file

        if not os.path.exists(self._input_file):
            raise FileNotFoundError(f"File {self._input_file} not found")

        self._output_dir = output_dir

        if not os.path.exists(self._output_dir):
            os.makedirs(self._output_dir)

        self._tagger = Tagger()
        self._tagger.load_model()

        self._llm = LLM()
        self._batch_size = batch_size

        self._downloader = DownloaderUrl()
        self._feature_extractor = FeatureExtractor()

    def download_from_file(
        self, file_path: str, batch_dl: int = 100, max_dl: int = -1
    ) -> None:
        """Download from file.

        Args:
            file_path (str): the file path that contain the urls.
            batch_dl (int, optional): the number of download in the same time. Defaults to 100.
            max_dl (int, optional): the maximum number of download. Defaults to -1.
        """

        with open(file_path, "r", encoding="utf-8") as f:
            urls = []
            for line in f:
                urls.append(line.split(";")[0].strip())

                if len(urls) < batch_dl:
                    continue

                logging.info("Downloading %s", len(urls))
                asyncio.run(self._downloader.download_all(urls, self._output_dir))
                urls = []

                if max_dl > 0 and len(os.listdir(self._output_dir)) >= max_dl:
                    break

    def process_download(self) -> None:
        """Method to process the downloaded files."""

        downloaded_files = []
        files_list = list(os.listdir(self._output_dir))

        with open(self._input_file, "r", encoding="utf-8") as f:
            for line in f.readlines():

                url, metadatas = line.strip().split(";", 1)
                file_name = url.split("/")[-1].strip()

                if file_name in files_list:
                    downloaded_files.append(
                        (os.path.join(self._output_dir, file_name), metadatas)
                    )

        batch_files = []
        for file in downloaded_files:
            batch_files.append(file)

            if len(batch_files) < self._batch_size and file != downloaded_files[-1]:
                continue

            logging.info("Tagging %s", len(batch_files))
            try:
                tags = self._tagger.tag([file for file, _ in batch_files], max_batch=3)
            except soundfile.LibsndfileError:
                logging.error("Error while tagging")
                continue

            logging.info("Generating features")

            features = [
                self._feature_extractor.print_features(file) for file, _ in batch_files
            ]

            inputs = [
                [
                    values[0].split("/")[-1].replace(".mp3", ""),
                    values[1],
                    batch_files[idx][1],
                    features[idx],
                ]
                for idx, values in enumerate(tags.items())
            ]

            logging.info("Generating prompts")
            outputs = self._llm.generate(inputs)

            for out in outputs:
                src.utils.save_prompt(
                    f"{self._output_dir}/{out['name']}.txt", out["description"]
                )

            batch_files = []

    def run(self) -> None:
        """
        Run the manager.
        """

        for filter_data in self.filters:
            if filter_data.type() == "text":
                filter_data.filter(self._input_file)

        # self.download_from_file(self._input_file, max_dl=5000)
        self.process_download()

        logging.info("Removing space data")
        for filter_data in self.filters:
            if filter_data.type() == "line":
                for file in os.listdir(self._output_dir):
                    if file.endswith(".txt"):
                        filter_data.filter(f"{self._output_dir}/{file}")

        logging.info("Done.")
