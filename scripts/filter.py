import argparse
import glob
import logging
import os

import eyed3
import tqdm

parser = argparse.ArgumentParser()
parser.add_argument(
    "--directory", help="directory path to filter music data", required=True
)
parser.add_argument(
    "--min_duration",
    help="minimum duration of audio file in seconds",
    type=int,
    default=3,
)
args = parser.parse_args()


def filter_audio_files(folder: str, min_duration: int = 3):
    # Remove invalid audio files
    for audio_file in tqdm.tqdm(glob.glob(os.path.join(folder, "*.mp3"))):
        path = os.path.join(folder, audio_file)
        file = eyed3.load(path)

        if file is None or file.info is None or file.info.time_secs < min_duration:
            logging.warning("Error processing audio file %s", path)
            os.remove(path)

            # remove metadata file
            metadata_file = path[:-4] + ".json"
            if os.path.exists(metadata_file):
                os.remove(metadata_file)


filter_audio_files(args.directory, args.min_duration)
