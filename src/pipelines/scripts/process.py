import argparse
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

from transcript import TranscriptModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)

# ARGUMENTS
parser = argparse.ArgumentParser()
parser.add_argument(
    "--output", help="output directory path", default="/media/works/audio_v2/"
)
parser.add_argument(
    "--workers", type=int, default=4, help="number of threads to use for transcription"
)
args = parser.parse_args()

model = TranscriptModel()


def process_audio_file(audio_file):
    if not audio_file.endswith(".mp3"):
        return None

    audio_path = os.path.join(args.output, audio_file)
    lyrics = model.transcript(audio_path)

    # Write transcript
    output_path = audio_path.rsplit(".", 1)[0] + "_transcript.txt"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(lyrics.strip())

    logging.info("Transcripted %s", audio_path)
    return audio_path


# Collect all .mp3 files
audio_files = [file for file in os.listdir(args.output) if file.endswith(".mp3")]

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
