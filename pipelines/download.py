import hashlib

from datasets import Dataset

from .downloaders import YoutubeDownloader


class Downloader:
    """Download audio from urls in the dataset"""

    def __init__(self, dataset: Dataset, cache_dir: str = ".pipelines"):
        self._dataset = dataset
        self._ytb_downloader = YoutubeDownloader(cache_dir=cache_dir)

        self._run()

    def _run(self):
        # add audio column to the dataset
        for split in self._dataset:
            for data in self._dataset[split]:
                url = data["url"]

                if url.startswith("https://www.youtube.com"):
                    base64_url = hashlib.sha256(url.encode("utf-8")).hexdigest()
                    self._ytb_downloader.add_url(url, base64_url)
