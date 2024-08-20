"""
Script to download a file from a URL
"""

import logging
import os
from typing import List

import asks
import trio
from asks.sessions import Session

from ..audio_codec import DAC
from .downloader_abc import Downloader


class DownloaderUrl(Downloader):
    """
    DownloaderUrl class using aiohttp for async downloads.
    """

    def __init__(self, max_dl_simultaneous=200):
        """DownloaderUrl constructor.

        Args:
            max_dl_simultaneous (int, optional): The maximum number of simultaneous downloads. Defaults to 200.
        """

        super().__init__()

        self._dl_simultaneous = 0
        self._max_dl_simultaneous = max_dl_simultaneous
        self._session = Session(connections=self._max_dl_simultaneous)
        self._audio_codec = DAC()

    def download_all(self, urls: List[str], output_dir: str):
        """
        Download multiple files from a list of URLs asynchronously.

        Args:
            urls (List[str]): The list of URLs to download (URL, metatags, file index).
            output_dir (str): The output directory.
            max_chunk (int, optional): The maximum number of chunks to download. Defaults to 1500.
        """

        async def grabber(url, metatags, path):
            end_task = False
            retry = 0
            while not end_task and retry <= 3:
                retry += 1

                try:
                    response = await self._session.get(url, stream=True)

                    if response.status_code == 200:
                        # get file size
                        if "Content-Length" not in response.headers:
                            logging.error("No Content-Length: %s", url)
                            end_task = True

                        if not end_task:
                            size = int(response.headers["Content-Length"])

                            # if bigger than 10MB or less than 5KB, skip
                            if (size > 10 * 1024 * 1024) or (size < 0.2 * 1024):
                                logging.error("File too big: %s", url)
                                end_task = True

                        if not end_task:
                            with open(os.path.join(output_dir, path), "wb") as f:
                                async for chunk in response.body(timeout=5):
                                    f.write(chunk)

                            # encode the audio
                            codes = self._audio_codec.encode(
                                os.path.join(output_dir, path)
                            )
                            self._audio_codec.save_tensor(
                                codes, os.path.join(output_dir, path[:-4] + ".pt")
                            )

                            # remove the mp3 file
                            os.remove(os.path.join(output_dir, path))

                            with open(
                                os.path.join(output_dir, path[:-4] + "_descr.txt"),
                                "w",
                                encoding="utf-8",
                            ) as f:
                                f.write(metatags)

                            end_task = True
                            logging.info("Downloaded: %s", url)
                except asks.errors.RequestTimeout:
                    logging.error("Timeout: %s", url)
                except OSError:
                    logging.error("OSError: %s", url)
                    self._session = Session(connections=self._max_dl_simultaneous)
                    await trio.sleep(60 * 3)

            self._dl_simultaneous -= 1

        async def main(urls):
            """Main function to download multiple files asynchronously.

            Args:
                urls (generator): The list of URLs to download.
            """

            async with trio.open_nursery() as n:
                for url, metatags, file_idx in urls:
                    self._dl_simultaneous += 1
                    n.start_soon(grabber, url, metatags, file_idx + ".mp3")

                    # wait un dl_simultaneous is less than 100
                    while self._dl_simultaneous > 100:
                        await trio.sleep(1)

        trio.run(main, urls)
