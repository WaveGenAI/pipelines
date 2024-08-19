import argparse
import logging
import os
import random

from downloader import DownloaderUrl

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)

# CONSTANTS
BATCH_DL_URLS = 500

# ARGUMENTS
parser = argparse.ArgumentParser()
parser.add_argument("--input", help="input file path", default="musics.xml")
parser.add_argument(
    "--output", help="output directory path", default="/media/works/audio_v3/"
)
args = parser.parse_args()

if not os.path.exists(args.input):
    raise FileNotFoundError(f"Input file not found: {args.input}")

os.makedirs(args.output, exist_ok=True)

downloader = DownloaderUrl()


def generator_urls():
    """Generator for URLs.

    Yields:
        tuple: URL, metatags, file index
    """

    # shuffle the file (to avoid downloading files to a single server)
    with open(args.input, "r", encoding="utf-8") as f:
        lines = f.readlines()
    random.shuffle(lines)
    with open(args.input, "w", encoding="utf-8") as f:
        f.writelines(lines)

    with open(args.input, "r", encoding="utf-8") as f:
        for line in f.readlines():
            urls, metatags = line.rsplit(";", 1)
            urls = urls.strip()
            metatags = metatags.strip()

            # create random id to name the file (fill with 0 if smaller than 10)
            file_idx = str(random.randint(0, 10**10)).zfill(10)
            while os.path.exists(os.path.join(args.output, file_idx + ".mp3")):
                file_idx = str(random.randint(0, 10**10)).zfill(10)

            yield urls, metatags, file_idx


downloader.download_all(generator_urls(), args.output)
