import argparse
import glob
import json
import logging
import os

import tqdm

from pipelines.transcript.transcription import TranscriptModel

args = argparse.ArgumentParser()
args.add_argument("--directory", type=str, required=True)
args = args.parse_args()

# disable logging info
logging.getLogger().setLevel(logging.WARNING)

asr = TranscriptModel()
BASE_DIR = args.directory

total_to_transcribe = 0

audio_file = glob.glob(BASE_DIR + "*.mp3")
# Get all audio that contains lyrics
for audio_file in tqdm.tqdm(audio_file, desc="Detect lyrics", total=len(audio_file)):
    base_name = os.path.basename(audio_file).rsplit(".", 1)[0]

    if not os.path.exists(os.path.join(BASE_DIR, base_name + ".json")):
        continue

    with open(os.path.join(BASE_DIR, base_name + ".json"), "r", encoding="utf-8") as f:
        metadata = json.load(f)

    if "lyrics" in metadata:
        if metadata["lyrics"] != "":
            total_to_transcribe += 1
        continue

    metadata["lyrics"] = ""
    audio_path = os.path.join(BASE_DIR, audio_file)
    try:
        is_lyrics = asr.contain_lyrics(audio_path)
        metadata["lyrics"] = "" if not is_lyrics else "TO BE FILLED"
    except RuntimeError as e:
        print(f"Error: {audio_path}, {e}")

    with open(os.path.join(BASE_DIR, base_name + ".json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f)

# reduce vram usage
del asr.validation_model

for audio_file in tqdm.tqdm(
    glob.glob(BASE_DIR + "*.mp3"), desc="Transcribe lyrics", total=total_to_transcribe
):
    base_name = os.path.basename(audio_file).rsplit(".", 1)[0]

    if not os.path.exists(os.path.join(BASE_DIR, base_name + ".json")):
        continue

    with open(os.path.join(BASE_DIR, base_name + ".json"), "r", encoding="utf-8") as f:
        metadata = json.load(f)

    if "lyrics" not in metadata:
        continue

    if metadata["lyrics"] != "TO BE FILLED":
        continue

    audio_path = os.path.join(BASE_DIR, audio_file)
    lyrics = asr.transcript(audio_path, check_lyrics=False)

    if lyrics != "":
        print(lyrics)
        print(audio_path)

    metadata["lyrics"] = lyrics

    with open(os.path.join(BASE_DIR, base_name + ".json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f)
