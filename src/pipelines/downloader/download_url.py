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
            urls (List[str]): The list of URLs to download (URL, metatags, file index).
            output_dir (str): The output directory.
            max_chunk (int, optional): The maximum number of chunks to download. Defaults to 1500.
        """

        async def grabber(s, url, path):
            try:
                response = await s.get(url, stream=True)

                if response.status_code == 200:
                    # get file size
                    if "Content-Length" not in response.headers:
                        logging.error("No Content-Length: %s", url)
                        return

                    size = int(response.headers["Content-Length"])

                    # if bigger than 10MB, skip
                    if size > 10 * 1024 * 1024:
                        logging.error("File too big: %s", url)
                        return

                    with open(os.path.join(output_dir, path), "wb") as f:
                        async for chunk in response.body(timeout=5):
                            f.write(chunk)

                    logging.info("Downloaded: %s", url)
            except asks.errors.RequestTimeout:
                logging.error("Timeout: %s", url)

        async def main(urls):
            session = Session(connections=len(urls))
            async with trio.open_nursery() as n:
                for url, _, file_idx in urls:
                    n.start_soon(grabber, session, url, file_idx + ".mp3")

        trio.run(main, urls)
