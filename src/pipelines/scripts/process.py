import argparse
import logging
import os

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
args = parser.parse_args()

model = TranscriptModel()

for audio_file in os.listdir(args.output):
    if not audio_file.endswith(".mp3"):
        continue

    audio_path = os.path.join(args.output, audio_file)
    lyrics = model.transcript(audio_path)

    # Write transcript
    with open(
        audio_path.rsplit(".")[0] + "_transcript.txt", "w", encoding="utf-8"
    ) as f:
        f.write(lyrics.strip())

    logging.info("Transcripted %s", audio_file)
