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

    def __init__(self, max_dl_simultaneous=100):
        """DownloaderUrl constructor.

        Args:
            max_dl_simultaneous (int, optional): The maximum number of simultaneous downloads. Defaults to 100.
        """

        super().__init__()

        self._dl_simultaneous = 0
        self._max_dl_simultaneous = max_dl_simultaneous

    def download_all(self, urls: List[str], output_dir: str):
        """
        Download multiple files from a list of URLs asynchronously.

        Args:
            urls (List[str]): The list of URLs to download (URL, metatags, file index).
            output_dir (str): The output directory.
            max_chunk (int, optional): The maximum number of chunks to download. Defaults to 1500.
        """

        async def grabber(s, url, metatags, path):
            self._dl_simultaneous += 1

            try:
                response = await s.get(url, stream=True)

                dl_file = True
                if response.status_code == 200:
                    # get file size
                    if "Content-Length" not in response.headers:
                        logging.error("No Content-Length: %s", url)
                        dl_file = False

                    size = int(response.headers["Content-Length"])

                    # if bigger than 10MB, skip
                    if size > 10 * 1024 * 1024:
                        logging.error("File too big: %s", url)
                        dl_file = False

                    if dl_file:
                        with open(os.path.join(output_dir, path), "wb") as f:
                            async for chunk in response.body(timeout=5):
                                f.write(chunk)

                        with open(
                            os.path.join(output_dir, path[:-4] + "_descr.txt"),
                            "w",
                            encoding="utf-8",
                        ) as f:
                            f.write(metatags)

                        logging.info("Downloaded: %s", url)
            except asks.errors.RequestTimeout:
                logging.error("Timeout: %s", url)

            self._dl_simultaneous -= 1
            return

        async def main(urls):
            """Main function to download multiple files asynchronously.

            Args:
                urls (generator): The list of URLs to download.
            """

            session = Session(connections=self._max_dl_simultaneous)
            async with trio.open_nursery() as n:
                for url, metatags, file_idx in urls:
                    n.start_soon(grabber, session, url, metatags, file_idx + ".mp3")

                    # wait un dl_simultaneous is less than 100
                    while self._dl_simultaneous > 100:
                        await trio.sleep(1)

        trio.run(main, urls)
