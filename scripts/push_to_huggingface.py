import argparse
import glob
import json
import os

import datasets
import torch

parser = argparse.ArgumentParser()
parser.add_argument("--directory", type=str, required=True)*
parser.add_argument("--repo", type=str, required=True)
args = parser.parse_args()

BASE_DIR = os.path.join(args.directory, "")


def gen_data():
    for audio_file in glob.glob(BASE_DIR + "*.pt"):
        # Check if the corresponding JSON file exists
        json_file = os.path.join(BASE_DIR, audio_file[:-3] + ".json")
        if not os.path.exists(json_file):
            continue

        with open(json_file, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        # Load the tensor in the pt file
        tensor = torch.load(audio_file)

        yield {
            "codec": tensor,  # tensor of shape [2, 9, seq_len] because using DAC (like Encodec)
            "prompt": metadata["prompt"],  # text
            "lyrics": metadata["lyrics"],  # text
        }


dataset = datasets.Dataset.from_generator(gen_data)
dataset.push_to_hub(args.repo, private=True)
