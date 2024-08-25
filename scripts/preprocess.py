import argparse
import glob
import json
import logging
import os
import random

import tqdm

args = argparse.ArgumentParser()
args.add_argument("--directory", type=str, required=True)
args = args.parse_args()

# disable logging info
logging.getLogger().setLevel(logging.WARNING)

BASE_DIR = args.directory
metatags_file = glob.glob(BASE_DIR + "*.json")

# Get all audio that contains lyrics
for metadata in tqdm.tqdm(
    metatags_file, desc="Creating prompts", total=len(metatags_file)
):
    with open(os.path.join(BASE_DIR, metadata), "r", encoding="utf-8") as f:
        data = json.load(f)

    tags = data["metatags"].split(",")
    tags = [tag.strip() for tag in tags]

    # remove tags that end with ":"
    tags = [tag for tag in tags if not tag.endswith(":")]

    # truncate the tag to remove first word that end with ":"
    tags = [tag.split(":")[1].strip() if ":" in tag else tag for tag in tags]

    # shuffle the tags
    random.shuffle(tags)
    data["prompt"] = " ".join(tags)

    with open(os.path.join(BASE_DIR, metadata), "w", encoding="utf-8") as f:
        json.dump(data, f)
