"""
Script to download a file from a URL
"""

import logging
import os
from typing import List

import asks
import trio
from asks.sessions import Session

from .downloader_abc import Downloader


class DownloaderUrl(Downloader):
    """
    DownloaderUrl class using aiohttp for async downloads.
    """

    def download_all(self, urls: List[str], output_dir: str):
        """
        Download multiple files from a list of URLs asynchronously.

        Args:
            urls (List[str]): List of URLs to download.
            output_dir (str): The output directory.
            max_chunk (int, optional): The maximum number of chunks to download. Defaults to 1500.
        """

        async def grabber(s, path):
            try:
                response = await s.get(path, stream=True, timeout=5)

                if response.status_code == 200:
                    # get file size
                    size = int(response.headers["Content-Length"])

                    # if bigger than 10MB, skip
                    if size > 10 * 1024 * 1024:
                        logging.error("File too big: %s", path)
                        return

                    with open(os.path.join(output_dir, path.split("/")[-1]), "wb") as f:
                        async for chunk in response.body(timeout=5):
                            f.write(chunk)

                    logging.info("Downloaded: %s", path)
            except asks.errors.RequestTimeout:
                logging.error("Timeout: %s", path)

        async def main(urls):
            session = Session(connections=len(urls))
            async with trio.open_nursery() as n:
                for url in urls:
                    n.start_soon(grabber, session, url)

        trio.run(main, urls)
