"""
Script to download a file from a URL
"""

import logging
import os

import requests

from src.downloader.downloader import Downloader
from src.exceptions import DownloadUrlException


class DownloaderUrl(Downloader):
    """
    DownloaderUrl class.
    """

    def download(self, url: str, output_dir: str, max_chunk: int = 1500) -> str:
        """
        Download the file from the URL.

        Args:
            url (str): The URL to download.
            output_dir (str): The output directory.
            max_chunk (int, optional): The maximum number of chunks to download. Defaults to 100.

        Returns:
            str: The path to the downloaded file.
        """

        base_filename = url.split("/")[-1]

        try:
            with requests.get(url, stream=True, timeout=10) as r:
                r.raise_for_status()

                # get if content type is audio
                content_type = r.headers.get("Content-Type")

                if content_type is None or "audio" not in content_type:
                    raise DownloadUrlException(f"Not an audio: {url}")

                if "Content-Length" not in r.headers:
                    raise DownloadUrlException(f"Content-Length not found: {url}")

                # if the file is too large (bigger than 5 minutes)
                if "Content-Length" in r.headers:
                    content_length = int(r.headers["Content-Length"])
                    if content_length > max_chunk * 8192:
                        raise DownloadUrlException(
                            f"File is too large: {content_length} bytes"
                        )

                with open(os.path.join(output_dir, base_filename), "wb") as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)

        except requests.exceptions.RequestException as e:
            raise DownloadUrlException(f"Failed to download URL: {url} - {e}")

        return os.path.join(output_dir, base_filename)
