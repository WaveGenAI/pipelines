""" 
Manager module.
"""

import asyncio
import logging
import os
import re

import soundfile
from lyric_whisper import LyricGen
from tagging.tagger import Tagger

import src.utils
from src.downloader import DownloaderUrl
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
        llm: bool = False,
        transcript: bool = False,
        download: bool = True,
    ):
        """Constructor for the Manager class.

        Args:
            input_file (str, optional): the input file that contain url. Defaults to "data.txt".
            output_dir (str, optional): the output directory. Defaults to "audio".
            batch_size (int, optional): the batch size. Defaults to 2.
            llm (bool, optional): Use the llm to generate the description. Defaults to False.
            transcript (bool, optional): Use the transcript to generate the description. Defaults to False.
            download (bool, optional): Download the audio files. Defaults to True.
        """

        self.filters = [DeduplicateFilter(), SpaceFilter()]
        self._input_file = input_file

        if not os.path.exists(self._input_file):
            raise FileNotFoundError(f"File {self._input_file} not found")

        self._output_dir = output_dir

        if not os.path.exists(self._output_dir):
            os.makedirs(self._output_dir)

        self._use_llm = llm
        self._batch_size = batch_size

        if llm:
            self._tagger = Tagger()
            self._tagger.load_model()
            self._llm = LLM()

        self._transcript = transcript

        if transcript:
            self._lyric_gen = LyricGen()

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

    def _construct_prompt_llm(self, batch: list) -> None:
        """Construct the prompt with llm.

        Args:
            batch (list): the batch of files.
        """
        logging.info("Tagging %s", len(batch))
        try:
            tags = self._tagger.tag([file for file, _ in batch], max_batch=3)
        except soundfile.LibsndfileError:
            logging.error("Error while tagging")
            return

        logging.info("Generating features")

        features = [self._feature_extractor.print_features(file) for file, _ in batch]

        inputs = [
            [
                values[0].split("/")[-1].replace(".mp3", ""),
                values[1],
                batch[idx][1],
                features[idx],
            ]
            for idx, values in enumerate(tags.items())
        ]

        logging.info("Generating prompts")
        outputs = self._llm.generate(inputs)

        for out in outputs:
            src.utils.save_prompt(
                f"{self._output_dir}/{out['name']}_descr.txt", out["description"]
            )

    def construct_prompt(self, batch: list) -> None:
        """Construct the prompt with llm

        Args:
            batch (list): the batch of files.
        """

        for file, metadatas in batch:
            name = file.split("/")[-1].replace(".mp3", "")
            metadatas = f"name: {name}, {metadatas}"

            # remove all word that end with :
            regex = r"\b\w+:\s"
            metadatas = re.sub(regex, "", metadatas)

            metadatas = src.utils.shuffle_list(metadatas.split(","))
            metadatas = ", ".join(metadatas)

            src.utils.save_prompt(f"{self._output_dir}/{name}_descr.txt", metadatas)

    def construct_lyric(self, batch: list) -> None:
        """Construct the prompt with lyric.

        Args:
            batch (list): the batch of files.
        """

        for file, _ in batch:
            name = file.split("/")[-1].replace(".mp3", "")
            prob_lyrics, lyrics = self._lyric_gen.generate_lyrics(file)

            if prob_lyrics > 0.5:
                src.utils.save_prompt(f"{self._output_dir}/{name}_lyric.txt", lyrics)

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

            if self._transcript:
                self.construct_lyric(batch_files)

            if self._use_llm:
                self._construct_prompt_llm(batch_files)
            else:
                self.construct_prompt(batch_files)

            batch_files = []

    def run(self) -> None:
        """
        Run the manager.
        """

        for filter_data in self.filters:
            if filter_data.type() == "text":
                filter_data.filter(self._input_file)

        if self._downloader:
            self.download_from_file(self._input_file, batch_dl=400)

        self.process_download()

        logging.info("Removing space data")
        for filter_data in self.filters:
            if filter_data.type() == "line":
                for file in os.listdir(self._output_dir):
                    if file.endswith(".txt"):
                        filter_data.filter(f"{self._output_dir}/{file}")

        logging.info("Done.")
