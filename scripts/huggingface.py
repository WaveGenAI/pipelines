import argparse
import logging
import os

import eyed3
from datasets import Audio, load_dataset

parser = argparse.ArgumentParser()
parser.add_argument("--folder", help="folder path", default="/media/works/test/")
args = parser.parse_args()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)


def filter_audio_files(folder):
    # Remove invalid audio files
    for audio_file in os.listdir(folder):
        if not audio_file.endswith(".mp3"):
            continue

        path = os.path.join(args.folder, audio_file)
        file = eyed3.load(path)

        if file is None or file.info is None or file.info.time_secs < 1:
            logging.warning("Error processing audio file %s", audio_file)
            os.remove(path)


filter_audio_files(args.folder)

dataset = load_dataset("audiofolder", data_dir=args.folder).cast_column(
    "audio", Audio(mono=False, sampling_rate=44100)
)

dataset.push_to_hub("WaveGenAI/audios")
