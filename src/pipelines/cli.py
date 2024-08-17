import argparse
import logging
import os
import random

import librosa
import numpy as np
import torch
from downloader import DownloaderUrl
from panns_inference import AudioTagging, labels
from transformers import pipeline

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
    asr = pipeline(
        "automatic-speech-recognition",
        "distil-whisper/distil-large-v3",
        device_map="auto",
        torch_dtype=torch.float16,
    )

    audio_tagging = AudioTagging(checkpoint_path=None, device="cuda")

    for audio_file in os.listdir(args.output):
        if not audio_file.endswith(".mp3"):
            continue

        audio_path = os.path.join(args.output, audio_file)
        try:
            (audio, _) = librosa.core.load(audio_path, sr=32000, mono=True)
        except Exception as e:
            logging.error(f"Error: {e}")
            continue

        audio = audio[None, :]

        try:
            (clipwise_output, embedding) = audio_tagging.inference(audio)
        except RuntimeError as e:
            logging.error(f"Error: {e}")
            continue

        sorted_indexes = np.argsort(clipwise_output[0])[::-1]

        lyrics = ""
        speech = False
        for k in range(10):
            if np.array(labels)[sorted_indexes[k]].lower() == "speech":
                if clipwise_output[0][sorted_indexes[k]] >= 0.5:
                    speech = True
                    break

        if speech:  # Transcript if no speech detected
            logging.info(f"Audio found in {audio_file}")
            # Transcribe audio
            transcript = asr(
                audio_path, chunk_length_s=30, batch_size=10, return_timestamps=True
            )

            lyrics = transcript["text"]

        # Write transcript
        with open(
            audio_path.rsplit(".")[0] + "_transcript.txt", "w", encoding="utf-8"
        ) as f:
            f.write(lyrics.strip())

        logging.info(f"Transcripted {audio_file}")
