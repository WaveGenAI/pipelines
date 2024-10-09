import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor

import ffmpeg
from yt_dlp import YoutubeDL

from pipelines.utils import cut_audio

logging.basicConfig(level=logging.INFO)


class YoutubeDownloader:
    """Class to find and return URLs of Youtube videos based on search terms."""

    def __init__(
        self,
        num_processes: int = 30,
        cache_dir: str = ".pipelines",
        audio_duration: int = 60 * 10,
    ):
        self._num_processes = num_processes
        self._cache_dir = cache_dir
        self._audio_duration = audio_duration
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
        while not success:
            ydl_opts = {
                "format": "m4a/bestaudio/best",
                "postprocessors": [
                    {
                        "key": "FFmpegExtractAudio",
                        "preferredcodec": "m4a",
                    }
                ],
                "proxy": "http://127.0.0.1:3128",
                "outtmpl": os.path.join(self._cache_dir, f"{file_name}_.m4a"),
                "quiet": True,
                "noprogress": True,
            }

            with YoutubeDL(ydl_opts) as ydl:
                try:
                    ydl.download([url])

                    ffmpeg.input(
                        os.path.join(self._cache_dir, f"{file_name}_.m4a")
                    ).output(
                        os.path.join(self._cache_dir, f"{file_name}.mp3"), format="mp3"
                    ).global_args(
                        "-loglevel", "quiet"
                    ).run(
                        overwrite_output=True
                    )  # fix soundfile reading error

                    os.remove(os.path.join(self._cache_dir, f"{file_name}_.m4a"))

                    # cut the audio file
                    cut_audio(
                        os.path.join(self._cache_dir, f"{file_name}.mp3"),
                        self._audio_duration,
                    )

                    success = True
                except Exception as error:  # pylint: disable=broad-except
                    non_critical_messages = [
                        "Sign in to confirm you’re not a bot",
                    ]

                    if not any(msg in str(error) for msg in non_critical_messages):
                        self.logging.error("Error downloading video: %s", url)
                        self.logging.error(error)
                        break

        if success:
            self.logging.info("Downloaded video: %s", url)

    def add_url(self, url: str, file_name: str):
        """Add a URL to the list of URLs to download."""

        while len(self.futures) >= self._num_processes:
            time.sleep(0.1)
            self._manage_futures()

        self.futures.add(self.executor.submit(self._download_ytb_video, url, file_name))
