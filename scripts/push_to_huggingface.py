import argparse
import glob
import os

import datasets
from datasets import Audio

parser = argparse.ArgumentParser()
parser.add_argument("--input_dir", type=str, required=True)
parser.add_argument("--output_dataset", type=str, required=True)
args = parser.parse_args()

BASE_DIR = os.path.join(args.input_dir, "")


def gen_data():
    for audio_file in glob.glob(BASE_DIR + "*.mp3"):
        # Check if the corresponding JSON file exists
        prompt_file = os.path.join(BASE_DIR, audio_file.split("_")[0] + ".txt")
        if not os.path.exists(prompt_file):
            continue

        prompt = open(prompt_file, "r", encoding="utf-8").read().strip()
        pos = audio_file.split("_")[-1].split(".")[0]

        yield {"audio": audio_file, "prompt": prompt, "position": int(pos)}


dataset = datasets.Dataset.from_generator(gen_data)
audio_dataset = dataset.cast_column("audio", Audio(mono=False, sampling_rate=44100))

for data in audio_dataset:
    print(data)

audio_dataset.push_to_hub(args.output_dataset)
