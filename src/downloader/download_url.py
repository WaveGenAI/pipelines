"""
Script to download a file from a URL
"""

import asyncio
import logging
import os
from typing import List

import aiohttp

from src.downloader.downloader import Downloader
from src.exceptions import DownloadUrlException


class DownloaderUrl(Downloader):
    """
    DownloaderUrl class using aiohttp for async downloads.
    """

    async def download(self, url: str, output_dir: str, max_chunk: int = 1500) -> None:
        """
        Download the file from the URL asynchronously.

        Args:
            url (str): The URL to download.
            output_dir (str): The output directory.
            max_chunk (int, optional): The maximum number of chunks to download. Defaults to 1500.
        """
        base_filename = url.split("/")[-1]

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as r:
                    if r.status != 200:
                        raise DownloadUrlException(
                            f"Failed to download URL: {url} - Status: {r.status}"
                        )

                    content_type = r.headers.get("Content-Type")
                    if content_type is None or "audio" not in content_type:
                        raise DownloadUrlException(f"Not an audio: {url}")

                    if "Content-Length" not in r.headers:
                        raise DownloadUrlException(f"Content-Length not found: {url}")

                    content_length = int(r.headers["Content-Length"])
                    if content_length > max_chunk * 8192:
                        raise DownloadUrlException(
                            f"File is too large: {content_length} bytes"
                        )

                    file_path = os.path.join(output_dir, base_filename)
                    with open(file_path, "wb") as f:
                        while True:
                            chunk = await r.content.read(8192)
                            if not chunk:
                                break
                            f.write(chunk)

                    logging.info("Downloaded %s", file_path)
        except DownloadUrlException as e:
            logging.error(e)

    async def download_all(
        self, urls: List[str], output_dir: str, max_chunk: int = 1500
    ):
        """
        Download multiple files from a list of URLs asynchronously.

        Args:
            urls (List[str]): List of URLs to download.
            output_dir (str): The output directory.
            max_chunk (int, optional): The maximum number of chunks to download. Defaults to 1500.
        """
        os.makedirs(output_dir, exist_ok=True)

        tasks = [self.download(url, output_dir, max_chunk) for url in urls]
        return await asyncio.gather(*tasks)
