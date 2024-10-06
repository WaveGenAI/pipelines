import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor

import ffmpeg
import pytubefix.exceptions
from pytubefix import YouTube

logging.basicConfig(level=logging.INFO)


class YoutubeDownloader:
    """Class to find and return URLs of Youtube videos based on search terms."""

    def __init__(
        self,
        num_processes: int = 80,
        cache_dir: str = ".pipelines",
    ):
        self._num_processes = num_processes
        self._cache_dir = cache_dir
        self.logging = logging.getLogger(__name__)

        # Create a thread pool with max 10 threads
        self.executor = ThreadPoolExecutor(max_workers=num_processes)
        self.futures = set()

    def _manage_futures(self):
        """Helper function to clean up completed futures and maintain a max of 10 threads."""
        # Check if any threads have finished and remove them
        completed_futures = {fut for fut in self.futures if fut.done()}
        for fut in completed_futures:
            fut.result()
            self.futures.remove(fut)

    def _download_ytb_video(self, url: str, file_name: str):
        if os.path.exists(os.path.join(self._cache_dir, f"{file_name}.mp3")):
            return

        success = False
        error_req = False
        while not success and not error_req:
            try:
                video = YouTube(
                    url,
                    proxies={
                        "http": "http://127.0.0.1:3128",
                        "https": "http://127.0.0.1:3128",
                    },
                )
                audio = video.streams.get_audio_only()
                audio.download(
                    mp3=True, output_path=self._cache_dir, filename=file_name + "_"
                )

                ffmpeg.input(os.path.join(self._cache_dir, f"{file_name}_.mp3")).output(
                    os.path.join(self._cache_dir, f"{file_name}.mp3"), format="mp3"
                ).global_args("-loglevel", "quiet").run(
                    overwrite_output=True
                )  # fix soundfile reading error

                os.remove(os.path.join(self._cache_dir, f"{file_name}_.mp3"))

                success = True
            except Exception as error:  # pylint: disable=broad-except
                non_critical_errors = [
                    pytubefix.exceptions.BotDetection,
                    pytubefix.exceptions.VideoUnavailable,
                ]
                non_critical_messages = [
                    "connection has been closed",
                    "Remote end closed connection without response",
                    "error Tunnel connection failed",
                ]

                if error.__class__ not in non_critical_errors and not any(
                    msg in str(error) for msg in non_critical_messages
                ):
                    self.logging.error("Error downloading video: %s", url)
                    self.logging.error(error)
                    error_req = True

        if success:
            self.logging.info("Downloaded video: %s", url)

    def add_url(self, url: str, file_name: str):
        """Add a URL to the list of URLs to download."""

        while len(self.futures) >= self._num_processes:
            time.sleep(0.1)
            self._manage_futures()

        self.futures.add(self.executor.submit(self._download_ytb_video, url, file_name))
