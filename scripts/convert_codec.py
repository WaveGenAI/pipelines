import argparse
import glob
import os

import dac
import torch
import tqdm
from accelerate import Accelerator
from audiotools import AudioSignal
from datasets import IterableDataset
from torch.utils.data import DataLoader

args = argparse.ArgumentParser()
args.add_argument("--directory", type=str, required=True)
args.add_argument("--delete", action="store_true")
args = args.parse_args()

BASE_DIR = os.path.join(args.directory, "")
audio_files = glob.glob(BASE_DIR + "*.mp3")

accelerator = Accelerator()
device = accelerator.device

model_path = dac.utils.download(model_type="44khz")
model = dac.DAC.load(model_path)
model.eval()
model.to(device)


def gen_data():
    for audio_file in audio_files:
        base_file = audio_file.split("/")[-1].split(".")[0]

        signal = AudioSignal(os.path.join(BASE_DIR, audio_file))
        signal = signal.resample(model.sample_rate)
        signal.to(model.device)

        yield signal.audio_data, signal.sample_rate, int(base_file)


ds = IterableDataset.from_generator(gen_data)
ds = ds.with_format("torch")

dataloader = DataLoader(ds)
dataloader = accelerator.prepare(dataloader)

batch = []
for data in tqdm.tqdm(dataloader, total=len(audio_files)):
    audio, sr, file_name = data
    x = model.preprocess(data[0], sr[0]).squeeze(0)

    x = x.transpose(0, 1)
    with torch.no_grad():
        _, c, _, _, _ = model.encode(x)

    # save the codes to a file
    batch.append((c, file_name.cpu().item()))

    if len(batch) <= 10:
        continue

    for c, file_name in batch:
        # save the batch in an array file
        path = os.path.join(BASE_DIR, f"{file_name}.pt")
        torch.save(c, path)

        # delete the audio file
        if args.delete:
            os.remove(os.path.join(BASE_DIR, f"{file_name}.mp3"))

    batch = []
