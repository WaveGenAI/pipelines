import argparse
import logging
import os
import random

import numpy as np
from downloader import DownloaderUrl
from faster_whisper import WhisperModel
from utils import *

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)

# CONSTANTS
BATCH_DL_URLS = 500
TRANSCRIPT_THRESHOLD = 0.6

# ARGUMENTS
parser = argparse.ArgumentParser()
parser.add_argument("--input", help="input file path", default="musics.xml")
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

    batch_urls = []
    with open(args.input, "r", encoding="utf-8") as f:
        for line in f.readlines():
            urls, metatags = line.rsplit(";", 1)
            urls = urls.strip()
            metatags = metatags.strip()

            # create random id to name the file (fill with 0 if smaller than 10)
            file_idx = str(random.randint(0, 10**10)).zfill(10)
            while os.path.exists(os.path.join(args.output, file_idx + ".mp3")):
                file_idx = str(random.randint(0, 10**10)).zfill(10)

            # stack urls while not reaching the limit
            batch_urls.append((urls, metatags, file_idx))

            if len(batch_urls) == BATCH_DL_URLS:
                downloader.download_all(batch_urls, args.output)

                for url, metatags, file_idx in batch_urls:
                    file_path = os.path.join(args.output, file_idx + ".mp3")

                    # skip if file not found (error during download)
                    if not os.path.exists(file_path):
                        continue

                    os.path.join(args.output, file_idx + "_descr.txt")
                    # write metatags
                    with open(
                        os.path.join(args.output, file_idx + "_descr.txt"),
                        "w",
                        encoding="utf-8",
                    ) as f:
                        f.write(metatags)

                batch_urls = []

if args.transcript:
    # Initialize ASR model
    model = WhisperModel("distil-large-v3", device="cuda", compute_type="float16")

    for audio_file in os.listdir(args.output):
        if not audio_file.endswith(".mp3"):
            continue

        audio_path = os.path.join(args.output, audio_file)

        # Transcribe audio file
        segments, info = model.transcribe(audio_path, beam_size=5)

        probs = []
        lyrics = ""

        for segment in segments:
            probs.append(segment.avg_logprob)

            if (
                np.exp(sum(probs) / len(probs)) < TRANSCRIPT_THRESHOLD
                and len(probs) > 5
            ):
                break

            lyrics += segment.text.strip() + "\n"

        if np.exp(sum(probs) / len(probs)) < TRANSCRIPT_THRESHOLD and len(probs) > 5:
            logging.warning("Low average logprob: %s", audio_file)
            lyrics = ""

        if evaluate_split_line(lyrics.strip()) < 4:
            logging.warning("Too short: %s", audio_file)
            lyrics = ""

        if lyrics != "":
            logging.info("Lyrics: %s", lyrics.strip())

        # Write transcript
        with open(
            audio_path.rsplit(".")[0] + "_transcript.txt", "w", encoding="utf-8"
        ) as f:
            f.write(lyrics.strip())

        logging.info("Transcripted %s", audio_file)
