"""
Script to download a file from a URL
"""

import logging
import os

import requests

from src.downloader.downloader import Downloader


class DownloaderUrl(Downloader):
    """
    DownloaderUrl class.
    """

    def download(self, url: str, output_dir: str) -> str:
        """
        Download the file from the URL.

        Args:
            url (str): The URL to download.
            output_dir (str): The output directory.

        Returns:
            str: The path to the downloaded file.
        """

        logging.info("Download URL: %s -> %s", url, output_dir)

        base_filename = url.split("/")[-1]

        with requests.get(url, stream=True, timeout=10) as r:
            r.raise_for_status()

            # get if content type is audio
            content_type = r.headers.get("Content-Type")

            if content_type is None or "audio" not in content_type:
                logging.error("URL is not an audio file.")
                return

            with open(os.path.join(output_dir, base_filename), "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)

        logging.info("Download URL done.")

        return os.path.join(output_dir, base_filename)
