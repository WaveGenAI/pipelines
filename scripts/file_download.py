import argparse
import csv
import json
import logging
import os
import random

from pipelines.downloader.download_url import DownloaderUrl

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)

# ARGUMENTS
parser = argparse.ArgumentParser()
parser.add_argument("--input_file", help="input csv file path", required=True)
parser.add_argument("--output_dir", help="output directory path", required=True)
args = parser.parse_args()

if not os.path.exists(args.input_file):
    raise FileNotFoundError(f"Input file not found: {args.input}")

os.makedirs(args.output_dir, exist_ok=True)

downloader = DownloaderUrl(100)


def generator_urls():
    """Generator for URLs.

    Yields:
        tuple: URL, file index
    """

    with open(args.input_file, "r", encoding="utf-8") as csvfile:
        reader = csv.reader(csvfile)
        header = next(reader)

        # find url header index
        url_idx = None
        for i, column in enumerate(header):
            if "url" in column.lower():
                url_idx = i
                break

        if url_idx is None:
            raise ValueError("URL column not found in the input file")

        for line in reader:
            if not line[url_idx].endswith(".mp3"):
                continue

            # create random id to name the file (fill with 0 if smaller than 10)
            file_idx = str(random.randint(0, 10**10)).zfill(10)
            while os.path.exists(os.path.join(args.output_dir, file_idx + ".mp3")):
                file_idx = str(random.randint(0, 10**10)).zfill(10)

            yield line[url_idx], file_idx


downloader.download_all(generator_urls(), args.output_dir)
