import argparse
import logging
import os
import tempfile

import webdataset as wds
from downloader.download_url import DownloaderUrl
from lyric_whisper import LyricGen

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)

# CONSTANTS
BATCH_DL_URLS = 1000

parser = argparse.ArgumentParser()
parser.add_argument("--input", help="input file path", default="../../musics.xml")
parser.add_argument(
    "--output", help="output directory path", default="/media/works/audio_v2/"
)
parser.add_argument("--llm", help="Pass data to llm", action="store_true")
parser.add_argument("--transcript", help="Pass data to transcript", action="store_true")
parser.add_argument("--download", help="Download audio files", action="store_true")
args = parser.parse_args()

if not os.path.exists(args.input):
    raise FileNotFoundError(f"Input file not found: {args.input}")

os.makedirs(args.output, exist_ok=True)

if args.download:
    downloader = DownloaderUrl()

    shard_id = 0
    tar_writer = wds.TarWriter(f"{args.output}/dataset-{shard_id:06d}.tar")

    batch_urls = []
    with open(args.input, "r", encoding="utf-8") as f:
        for line in f.readlines():
            urls, metatags = line.rsplit(";", 1)
            name = urls.split("/")[-1].strip()
            urls = urls.strip()
            metatags = metatags.strip()

            # stack urls while not reaching the limit
            batch_urls.append((urls, metatags, name))

            if len(batch_urls) == BATCH_DL_URLS:
                with tempfile.TemporaryDirectory() as tmpdirname:
                    downloader.download_all(
                        [url for url, _, _ in batch_urls], tmpdirname
                    )

                    for url, metatags, name in batch_urls:
                        file_path = os.path.join(tmpdirname, url.split("/")[-1])

                        # skip if file not found (error during download)
                        if not os.path.exists(file_path):
                            continue

                        with open(file_path, "rb") as audio_file:
                            tar_writer.write(
                                {
                                    "__key__": name,
                                    "wav": audio_file.read(),
                                    "txt": metatags.encode("utf-8"),
                                }
                            )
                        logging.info("Saved to tar: %s", name)

                        # remove downloaded audio file
                        os.remove(file_path)

                shard_id += 1
                tar_writer = wds.TarWriter(f"{args.output}/dataset-{shard_id:06d}.tar")
                batch_urls = []

    if batch_urls:
        downloader = DownloaderUrl()
        downloaded_files = downloader.download_all(
            [url for url, _, _ in batch_urls], args.output
        )

        # Sauvegarder les fichiers restants dans le tarball
        for (url, metatags, name), file_path in zip(batch_urls, downloaded_files):
            with open(file_path, "rb") as audio_file:
                tar_writer.write(
                    {
                        "__key__": name,
                        "wav": audio_file.read(),
                        "txt": metatags.encode("utf-8"),
                    }
                )
            logging.info("Saved to tar: %s", name)

    # Fermer le writer à la fin
    tar_writer.close()
