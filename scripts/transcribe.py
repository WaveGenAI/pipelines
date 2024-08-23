import logging

import soundfile
from audiotools import AudioSignal
from datasets import Audio, load_dataset

from pipelines.transcript import TranscriptModel

asr = TranscriptModel()
ds = load_dataset(
    "audiofolder", data_dir="/media/works/test/", split="train"
).cast_column("audio", Audio(mono=False))

# select first 50 samples
ds = ds.select(range(30_000))


def prepare_audio(row, idx):
    audio, sr = ds[idx]["audio"]["array"], ds[idx]["audio"]["sampling_rate"]
    audio = AudioSignal(audio, sr)

    # convert to mono and resample
    audio = audio.to_mono()
    audio = audio.resample(16000)

    try:
        transcription = asr.transcript(audio.numpy()[0, 0])
        row["lyrics"] = transcription
    except soundfile.LibsndfileError:
        row["lyrics"] = ""
        logging.warning("Error processing audio file %s", idx)

    return row


ds = ds.map(prepare_audio, with_indices=True)
ds.save_to_disk("/media/works/test/lyrics")

# push to hub
ds.push_to_hub("WaveGenAI/lyrics", private=True)
