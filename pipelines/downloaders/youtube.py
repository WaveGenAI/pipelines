import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor

import pytubefix.exceptions
from pydub import AudioSegment
from pytubefix import YouTube

logging.basicConfig(level=logging.INFO)


class YoutubeDownloader:
    """Class to find and return URLs of Youtube videos based on search terms."""

    def __init__(
        self,
        num_processes: int = 40,
    ):
        self._num_processes = num_processes
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
        if os.path.exists(f".pipelines/{file_name}.mp3"):
            return

        success = False
        while not success:
            try:
                video = YouTube(
                    url,
                    proxies={
                        "http": "http://127.0.0.1:3128",
                        "https": "http://127.0.0.1:3128",
                    },
                )
                audio = video.streams.get_audio_only()
                audio.download(mp3=True, output_path=".pipelines", filename=file_name)

                # convert to wav
                sound = AudioSegment.from_file(f".pipelines/{file_name}.mp3")
                sound.export(f".pipelines/{file_name}.wav", format="wav")

                # delete mp3
                os.remove(f".pipelines/{file_name}.mp3")

                success = True
            except Exception as error:  # pylint: disable=broad-except
                if error.__class__ not in (
                    pytubefix.exceptions.BotDetection,
                    pytubefix.exceptions.VideoUnavailable,
                ):
                    self.logging.error("Error downloading video: %s", url)
                    self.logging.error(error)

        self.logging.info("Downloaded video: %s", url)

    def add_url(self, url: str, file_name: str):
        """Add a URL to the list of URLs to download."""

        while len(self.futures) >= self._num_processes:
            time.sleep(0.1)
            self._manage_futures()

        self.futures.add(self.executor.submit(self._download_ytb_video, url, file_name))
