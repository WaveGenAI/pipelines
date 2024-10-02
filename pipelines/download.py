from datasets import Dataset

from .downloaders import YoutubeDownloader


class Downloader:
    """Download audio from urls in the dataset"""

    def __init__(self, dataset: Dataset):
        self._dataset = dataset
        self._ytb_downloader = YoutubeDownloader()

        self._run()

    def _run(self):
        for column in self._dataset:
            for data in self._dataset[column]:
                url = data["url"]

                if url.startswith("https://www.youtube.com"):
                    self._ytb_downloader.add_url(url)
