import argparse
import glob
import os

import dac
import torch
from datasets import Audio, Dataset, DatasetDict, load_dataset

parser = argparse.ArgumentParser()
parser.add_argument(
    "--input_dataset",
    type=str,
    required=True,
    help="Either a Hugging Face dataset name or a local directory",
)
parser.add_argument(
    "--streaming",
    action="store_true",
    help="Whether to use the streaming version of the dataset",
)
parser.add_argument("--output_dataset", type=str, required=True)
args = parser.parse_args()

# Download the model
model_path = dac.utils.download(model_type="44khz")
model = dac.DAC.load(model_path)
model.to("cuda")


def gen_data_from_directory(input_dir):
    """
    Generate data from a directory like `gen_data` in push_to_huggingface.py.
    """
    BASE_DIR = os.path.join(input_dir, "")

    def gen_data():
        for audio_file in glob.glob(BASE_DIR + "*.mp3"):
            # Check if the corresponding JSON file exists
            prompt_file = os.path.join(BASE_DIR, audio_file.split("_")[0] + ".txt")
            if not os.path.exists(prompt_file):
                continue

            prompt = open(prompt_file, "r", encoding="utf-8").read().strip()
            pos = audio_file.split("_")[-1].split(".")[0]

            yield {"audio": audio_file, "prompt": prompt, "position": int(pos)}

    return gen_data


# Load the dataset
if os.path.isdir(args.input_dataset):
    # Load the dataset from a directory
    train_dataset = Dataset.from_generator(gen_data_from_directory(args.input_dataset))
    train_dataset = train_dataset.cast_column(
        "audio", Audio(mono=False, sampling_rate=44100)
    )
    dataset = DatasetDict({"train": train_dataset})

else:
    # Load the dataset from the Hugging Face hub
    dataset = load_dataset(args.input_dataset, streaming=args.streaming)


@torch.no_grad()
def dataset_generator():
    """
    This function will be called by the Dataset.from_generator function.
    """
    # Iterate over the dataset splits
    for split in dataset:
        for data in dataset[split]:
            audio = torch.Tensor(data["audio"]["array"])

            # add channel dimension if needed (mono)
            if audio.dim() == 1:
                audio = audio.unsqueeze(0)

            # move to [C, T] to [C, 1, T]
            audio = audio.unsqueeze(1).to(model.device)
            x = model.preprocess(audio, data["audio"]["sampling_rate"])
            _, codes, _, _, _ = model.encode(x)

            # get the prompt
            prompt = data["prompt"]

            yield {
                "codes": codes.cpu().numpy(),
                "prompt": prompt,
            }


codec_dataset = Dataset.from_generator(dataset_generator)
codec_dataset.push_to_hub(args.output_dataset)
