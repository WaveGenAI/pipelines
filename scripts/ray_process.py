import os
from typing import Any, Dict

import datasets
import ray
from datasets import Audio

# load dataset from generator
BASE_DIR = "/media/works/audio_v3/"


def generate_data():
    for audio_file in os.listdir(BASE_DIR):
        if not audio_file.endswith(".mp3"):
            continue

        audio_path = os.path.join(BASE_DIR, audio_file)
        if not os.path.exists(audio_path[:-4] + "_descr.txt"):
            continue

        description = ""
        with open(audio_path.replace(".mp3", "_descr.txt"), "r", encoding="utf-8") as f:
            description = f.read()

        yield {
            "audio_path": audio_path,
            "description": description,
        }


data = datasets.Dataset.from_generator(generate_data)
data = data.cast_column("audio_path", Audio(mono=False, decode=True))
ds = ray.data.from_huggingface(data)


class AudioTranscriber:
    def __init__(self):
        self.model = None  # Delay model loading

    def __call__(self, batch: Dict[Any, str]) -> Dict[Any, str]:
        if self.model is None:
            from transcript import TranscriptModel

            self.model = TranscriptModel()

        batch["transcription"] = self.model.transcript(batch["audio_path"]["path"])

        return batch


# map dataset
ds = ds.map(AudioTranscriber(), num_gpus=1)

ds.write_parquet(BASE_DIR)
