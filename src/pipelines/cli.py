import argparse
import io
import logging
import os
import tempfile

import braceexpand
import torch
import torchaudio
import webdataset as wds
from downloader import DownloaderUrl
from silero_vad import get_speech_timestamps, load_silero_vad
from transformers import pipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)

# CONSTANTS
BATCH_DL_URLS = 500
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SAMPLE_RATE = 44100

# ARGUMENTS
parser = argparse.ArgumentParser()
parser.add_argument("--input", help="input file path", default="musics.xml")
parser.add_argument(
    "--output", help="output directory path", default="/media/works/audio_v2/"
)
parser.add_argument("--llm", help="Pass data to llm", action="store_true")
parser.add_argument("--transcript", help="Pass data to transcript", action="store_true")
parser.add_argument("--download", help="Download audio files", action="store_true")
args = parser.parse_args()

if not os.path.exists(args.input):
    raise FileNotFoundError(f"Input file not found: {args.input}")

os.makedirs(args.output, exist_ok=True)

if args.download:
    downloader = DownloaderUrl()

    shard_id = 0
    tar_writer = wds.TarWriter(f"{args.output}/dataset-{shard_id:06d}.tar")

    batch_urls = []
    with open(args.input, "r", encoding="utf-8") as f:
        for line in f.readlines():
            urls, metatags = line.rsplit(";", 1)
            name = urls.split("/")[-1].strip()
            urls = urls.strip()
            metatags = metatags.strip()

            # stack urls while not reaching the limit
            batch_urls.append((urls, metatags, name))

            if len(batch_urls) == BATCH_DL_URLS:
                with tempfile.TemporaryDirectory() as tmpdirname:
                    downloader.download_all(
                        [url for url, _, _ in batch_urls], tmpdirname
                    )

                    idx = 0
                    for url, metatags, name in batch_urls:
                        file_path = os.path.join(tmpdirname, url.split("/")[-1])

                        # skip if file not found (error during download)
                        if not os.path.exists(file_path):
                            continue

                        idx_str = f"{idx:06d}"
                        idx += 1

                        with open(file_path, "rb") as audio_file:
                            tar_writer.write(
                                {
                                    "__key__": idx_str,
                                    "wav": audio_file.read(),
                                    "txt": metatags.encode("utf-8"),
                                }
                            )
                        logging.info("Saved to tar: %s", idx_str)

                        # remove downloaded audio file
                        os.remove(file_path)

                shard_id += 1
                tar_writer = wds.TarWriter(f"{args.output}/dataset-{shard_id:06d}.tar")
                batch_urls = []

    if batch_urls:
        downloader = DownloaderUrl()
        downloaded_files = downloader.download_all(
            [url for url, _, _ in batch_urls], args.output
        )

        idx = 0
        for (url, metatags, name), file_path in zip(batch_urls, downloaded_files):
            with open(file_path, "rb") as audio_file:
                # fill idx with zeros
                idx_str = f"{idx:06d}"
                idx += 1

                tar_writer.write(
                    {
                        "__key__": idx_str,
                        "wav": audio_file.read(),
                        "txt": metatags.encode("utf-8"),
                    }
                )

            logging.info("Saved to tar: %s", idx_str)

    tar_writer.close()


num_tar_files = len(os.listdir("/media/works/audio_v2/"))
dataset_path = os.path.join(args.output, f"dataset-{{000000..{num_tar_files - 1}}}.tar")
urls = braceexpand.braceexpand(dataset_path)

dataset = wds.WebDataset(urls)


def resample_audio(sample: dict):
    audio_data, sr = torchaudio.load(io.BytesIO(sample["wav"]))

    if sr != SAMPLE_RATE:
        audio_data = torchaudio.transforms.Resample(orig_freq=sr, new_freq=SAMPLE_RATE)(
            audio_data
        )

    sample["wav"] = audio_data
    return sample


# resample audio
dataset = dataset.map(resample_audio)


vad = load_silero_vad()
vad.to(DEVICE)
asr = pipeline(
    "automatic-speech-recognition",
    "distil-whisper/distil-large-v3",
    device=DEVICE,
    torch_dtype=torch.float16,
)


# transcribe audio
def transcribe_audio(sample: dict):
    audio_data = sample["wav"]
    sample["lyrics"] = ""

    # convert to mono
    if audio_data.shape[0] > 1:
        audio_data = torch.mean(audio_data, dim=0, keepdim=False)

    audio_data = audio_data.to(DEVICE)

    # get speech timestamps
    speech_timestamps = get_speech_timestamps(audio_data, vad)

    if len(speech_timestamps) < 3:
        return sample

    outputs = asr(
        audio_data.cpu().numpy(),
        chunk_length_s=30,
        batch_size=10,
        return_timestamps=False,
    )

    sample["lyrics"] = outputs["text"].strip()

    return sample


dataset = dataset.map(transcribe_audio)

for sample in dataset:
    print(sample)
    pass
