import glob
import hashlib

from datasets import Dataset

from .downloaders import YoutubeDownloader


class Downloader:
    """Download audio from urls in the dataset"""

    def __init__(
        self, dataset: Dataset, cache_dir: str = ".pipelines", max_files: int = 0
    ):
        if not isinstance(max_files, int) or max_files < 0:
            raise ValueError(
                "max_files should be an integer greater than or equal to 0"
            )

        self._dataset = dataset
        self._cache_dir = cache_dir
        self._max_files = max_files
        self._ytb_downloader = YoutubeDownloader(cache_dir=cache_dir)

        self._run()

    def _run(self):
        # add audio column to the dataset
        for split in self._dataset:
            for data in self._dataset[split]:
                if (
                    self._max_files != 0
                    and len(glob.glob(f"{self._cache_dir}/*.mp3")) >= self._max_files
                ):
                    break

                url = data["url"]

                if url.startswith("https://www.youtube.com"):
                    base64_url = hashlib.sha256(url.encode("utf-8")).hexdigest()
                    self._ytb_downloader.add_url(url, base64_url)
