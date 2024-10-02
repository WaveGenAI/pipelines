from datasets import Dataset

from .downloaders import YoutubeDownloader


class Downloader:
    """Download audio from urls in the dataset"""

    def __init__(self, dataset: Dataset):
        self._dataset = dataset
        self._ytb_downloader = YoutubeDownloader()

        self._run()

    def _run(self):
        # add audio column to the dataset
        for split in self._dataset:
            for idx, data in enumerate(self._dataset[split]):
                url = data["url"]

                if url.startswith("https://www.youtube.com"):
                    self._ytb_downloader.add_url(url, f"{split}_{idx}")
