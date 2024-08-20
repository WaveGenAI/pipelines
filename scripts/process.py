import argparse
import logging
import os
import signal
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed

import torch

from pipelines.audio_codec import DAC
from pipelines.transcript import TranscriptModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)

# ARGUMENTS
parser = argparse.ArgumentParser()
parser.add_argument(
    "--output", help="output directory path", default="/media/works/audio_v3/"
)
parser.add_argument(
    "--workers", type=int, default=1, help="number of threads to use for transcription"
)
args = parser.parse_args()

model = TranscriptModel()
codec_audio = DAC()

# Global flag to stop threads
stop_threads = False


def process_audio_file(audio_file):
    if stop_threads:
        return
    audio_path = os.path.join(args.output, audio_file)

    # Check if transcript already exists
    output_path = audio_path.rsplit(".", 1)[0] + "_transcript.txt"
    if os.path.exists(output_path):
        logging.info("Already processed %s", audio_file)
        return

    with torch.no_grad():
        # load audio tensor
        codes = codec_audio.load_tensor(os.path.join(args.output, audio_file))
        waveform = codec_audio.decode(s=codes)

    # create temporary file
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as f:
        codec_audio.waveform_to_audiofile(waveform, f.name)
        lyrics = model.transcript(f.name)

    # Write transcript
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(lyrics.strip())


def signal_handler(sig, frame):
    global stop_threads
    logging.info("Received signal to stop processing...")
    stop_threads = True
    executor.shutdown(wait=False)
    sys.exit(0)


# Register signal handler for Ctrl+C
signal.signal(signal.SIGINT, signal_handler)

# Collect all .mp3 files
audio_files = [file for file in os.listdir(args.output) if file.endswith(".pt")]

# Use ThreadPoolExecutor to process files in parallel
with ThreadPoolExecutor(max_workers=args.workers) as executor:
    futures = {
        executor.submit(process_audio_file, audio_file): audio_file
        for audio_file in audio_files
    }

    for future in as_completed(futures):
        audio_file = futures[future]
        try:
            result = future.result()
            if result:
                logging.info("Successfully processed %s", result)
        except Exception as e:
            logging.error("Error processing %s: %s", audio_file, e)
