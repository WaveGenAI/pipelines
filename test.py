import concurrent
import math
import os
import time
from urllib.parse import quote

import litdata as ld
from lightning.data.processing.utilities import catch, make_request


def get_urls():
    with open("musics.xml", "r", encoding="utf-8") as f:
        for line in f.readlines():
            urls, metatags = line.rsplit(";", 1)
            urls = urls.strip()
            metatags = metatags.strip()

            yield {"url": urls, "metatags": metatags}


def download_audio_and_prepare(row):
    url, metatags = row["url"], row["metatags"]
    encoded_url = quote(url, safe=":/")

    out = make_request(encoded_url, timeout=1.5)
    data = {"audio": out, "description": metatags, "url": url}

    return data


class AudioFetcher:
    def __init__(self, max_threads=os.cpu_count()):
        self.max_threads = max_threads

    def initialize_executor(self):
        self.thread_pool = concurrent.futures.ThreadPoolExecutor(self.max_threads)

    def __call__(self, batch):
        self.initialize_executor()

        futures = [
            self.thread_pool.submit(catch(download_audio_and_prepare), row)
            for row in batch
        ]

        for future in concurrent.futures.as_completed(futures):
            data, err = future.result()

            if data is None:
                continue

            yield data


if __name__ == "__main__":
    data = list(get_urls())
    max_dl = 4000

    data = data[:max_dl]

    batch_size = 2048
    batch = [
        data[i * batch_size : (i + 1) * batch_size]
        for i in range(math.ceil(len(data) / batch_size))
    ]

    ld.optimize(
        fn=AudioFetcher(16),
        inputs=batch,
        output_dir="/media/works/test/",
        chunk_bytes="64MB",
        mode="overwrite",
        num_workers=min(os.cpu_count(), len(batch)),
    )
