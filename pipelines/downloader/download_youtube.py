"""
Script to download a file from a URL
"""

import csv
import os
from typing import List

import yt_dlp

from .downloader_abc import Downloader


class DownloadYoutube(Downloader):
    """
    DownloaderUrl class using aiohttp for async downloads.
    """

    def __init__(self, output_dir: str):
        """
        Initialize the DownloaderUrl class.

        Args:
            output_dir (str): The output directory.
        """

        super().__init__()
        self._batch_dl = 10
        self._output_dir = output_dir

    def _write_index(self, url, index: int):
        """Write the index to a file.

        Args:
            index (int): The index to write.
        """

        if not os.path.exists(os.path.join(self._output_dir, "index.csv")):
            with open(
                os.path.join(self._output_dir, "index.csv"),
                "w",
                newline="",
                encoding="utf-8",
            ) as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(["url", "index"])

        with open(
            os.path.join(self._output_dir, "index.csv"),
            "a",
            newline="",
            encoding="utf-8",
        ) as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([url, index])

    def _my_hook(self, d):
        if d["status"] == "finished":
            self._write_index(d["info_dict"]["original_url"], d["filename"])

    def _download_batch(self, urls: List[str]):
        ydl_opts = {
            "format": "mp3/bestaudio/best",
            "progress_hooks": [self._my_hook],
            "outtmpl": f"{self._output_dir}/%(id)s.%(ext)s",
            "postprocessors": [
                {
                    "key": "FFmpegExtractAudio",
                    "preferredcodec": "mp3",
                }
            ],
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download(urls)

    def download_all(self, urls: List[str]):
        """
        Download multiple files from a list of URLs asynchronously.
        """

        batch_urls = []
        for url in urls:
            batch_urls.append(url)
            if len(batch_urls) == self._batch_dl:
                self._download_batch(batch_urls)
                batch_urls = []
