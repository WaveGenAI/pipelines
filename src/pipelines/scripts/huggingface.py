import os

from datasets import Audio, Dataset
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--folder", help="folder path", default="/media/works/audio_v3")
args = parser.parse_args()

audio_files = []
descriptions = []

for file_name in os.listdir(args.folder):
    if file_name.endswith(".mp3"):
        if not os.path.exists(
            os.path.join(args.folder, file_name.replace(".mp3", "_descr.txt"))
        ):
            continue

        audio_files.append(os.path.join(args.folder, file_name))
        description_file = file_name.replace(".mp3", "_descr.txt")
        with open(
            os.path.join(args.folder, description_file), "r", encoding="utf-8"
        ) as desc_file:
            descriptions.append(desc_file.read())

data = {"audio": audio_files, "description": descriptions}
dataset = Dataset.from_dict(data)

dataset = dataset.cast_column("audio", Audio(mono=False, decode=False))

dataset.push_to_hub("Jour/audio")
print(dataset)
