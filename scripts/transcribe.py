import json
import logging
import os

import tqdm

from pipelines.transcript.transcription import TranscriptModel

# disable logging info
logging.getLogger().setLevel(logging.WARNING)

asr = TranscriptModel()
BASE_DIR = "/media/works/test2/"

total_to_transcribe = 0

# Get all audio that contains lyrics
for audio_file in tqdm.tqdm(
    os.listdir(BASE_DIR), desc="Detect lyrics", total=len(os.listdir(BASE_DIR)) // 2
):
    if not audio_file.endswith(".mp3"):
        continue

    base_name = os.path.basename(audio_file).rsplit(".", 1)[0]

    if not os.path.exists(os.path.join(BASE_DIR, base_name + ".json")):
        continue

    with open(os.path.join(BASE_DIR, base_name + ".json"), "r", encoding="utf-8") as f:
        metadata = json.load(f)

    if "lyrics" in metadata:
        if metadata["lyrics"] != "":
            total_to_transcribe += 1
        continue

    audio_path = os.path.join(BASE_DIR, audio_file)
    try:
        is_lyrics = asr.contain_lyrics(audio_path)
    except RuntimeError:
        continue

    metadata["lyrics"] = "" if not is_lyrics else "TO BE FILLED"

    with open(os.path.join(BASE_DIR, base_name + ".json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f)

del asr.validation_model

for audio_file in tqdm.tqdm(
    os.listdir(BASE_DIR), desc="Transcribe lyrics", total=total_to_transcribe
):
    if not audio_file.endswith(".mp3"):
        continue

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
