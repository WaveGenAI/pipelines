""" 
Manager module.
"""

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
        batch_size: int = 4,
    ):
        """Constructor for the Manager class.

        Args:
            input_file (str, optional): the input file that contain url. Defaults to "data.txt".
            output_dir (str, optional): the output directory. Defaults to "audio".
            batch_size (int, optional): the batch size. Defaults to 4.
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
                try:
                    tags = self._tagger.tag(
                        [file for file, _ in downloaded_files], max_batch=10
                    )
                except soundfile.LibsndfileError:
                    logging.error("Error while tagging")
                    continue

                logging.info("Generating features")

                features = [
                    self._feature_extractor.print_features(file)
                    for file, _ in downloaded_files
                ]

                lst_tags = [
                    [
                        values[0].split("/")[-1].replace(".mp3", ""),
                        values[1],
                        downloaded_files[idx][1],
                        features[idx],
                    ]
                    for idx, values in enumerate(tags.items())
                ]

                logging.info("Generating prompts")
                outputs = self._llm.generate(lst_tags)

                for out in outputs:
                    src.utils.save_prompt(
                        f"{self._output_dir}/{out['name']}.txt", out["description"]
                    )

        logging.info("Removing space data")
        for filter_data in self.filters:
            if filter_data.type() == "line":
                for file in os.listdir(self._output_dir):
                    if file.endswith(".txt"):
                        filter_data.filter(f"{self._output_dir}/{file}")

        logging.info("Done.")
