import argparse
import glob
import os

import dac
import soundfile as sf
import torch
from datasets import Audio, Dataset, DatasetDict, load_dataset

from pipelines.audio_mae import AudioMAEConfig, PretrainedAudioMAEEncoder

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
parser.add_argument(
    "--max_duration",
    type=int,
    default=60 * 5,
    help="Maximum duration of the audio file (chunks of 30 seconds)",
)
parser.add_argument(
    "--test_size",
    type=int,
    default=2000,
    help="Number of elements to put in the test dataset",
)
parser.add_argument("--output_dataset", type=str, required=True)
args = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"

# Download the model
model_path = dac.utils.download(model_type="44khz")
model = dac.DAC.load(model_path)
model.to(device)

config = AudioMAEConfig()
mae = PretrainedAudioMAEEncoder(config).load_model().to(device)


def gen_data_from_directory(input_dir):
    """
    Generate data from a directory like `gen_data` in push_to_huggingface.py.
    """
    BASE_DIR = os.path.join(input_dir, "")

    def gen_data():
        # precompute the number of chunks for each audio file
        nb_chunks_map = {}
        for file in glob.glob(os.path.join(BASE_DIR, "*.mp3")):
            prefix = file.split("_")[0]

            if prefix not in nb_chunks_map:
                nb_chunks_map[prefix] = 1
            else:
                nb_chunks_map[prefix] += 1

        for audio_file in glob.glob(BASE_DIR + "*.mp3"):
            # Check if the corresponding JSON file exists
            prompt_file = os.path.join(BASE_DIR, audio_file.split("_")[0] + ".txt")
            if not os.path.exists(prompt_file):
                continue

            pos = audio_file.split("_")[-1].split(".")[0]
            # Get the number of chunks
            nb_chunks = nb_chunks_map[audio_file.split("_")[0]]

            if int(pos) > 5:
                continue

            # try to open the audio file to check if it's valid
            try:
                _, _ = sf.read(audio_file)
            except (sf.LibsndfileError, ValueError):
                continue

            prompt = open(prompt_file, "r", encoding="utf-8").read().strip()

            yield {
                "audio": audio_file,
                "prompt": prompt,
                "chunk_id": int(pos),
                "nb_chunks": nb_chunks,
            }

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
def dataset_generator(skip: int = None, take: int = None):
    """
    This function will be called by the Dataset.from_generator function.
    """
    # Iterate over the dataset splits
    i = 0
    for split in dataset:
        for data in dataset[split]:
            if skip is not None and i <= skip:
                continue

            if take is not None and i >= take:
                return

            i += 1
            audio = torch.Tensor(data["audio"]["array"])

            embd = mae.forward(data["audio"]["path"])

            # add channel dimension if needed (mono)
            if audio.dim() == 1:
                audio = audio.unsqueeze(0)

            # move to [C, T] to [C, 1, T]
            audio = audio.unsqueeze(1).to(model.device)
            x = model.preprocess(audio, data["audio"]["sampling_rate"])
            _, codes, _, _, _ = model.encode(x)

            # remove audio from the data
            del data["audio"]

            yield {
                "codes": codes.long().cpu().numpy(),
                "embd": embd.cpu().numpy(),
                **data,
            }


# Push the dataset to the Hugging Face hub, retrying if it fails
while True:
    # Create the test dataset with 2000 elements
    test_dataset = Dataset.from_generator(
        lambda: dataset_generator(skip=None, take=args.test_size)
    )

    # Create the training dataset by skipping the first 2000 elements
    train_dataset = Dataset.from_generator(
        lambda: dataset_generator(skip=args.test_size, take=None)
    )

    # Combine the datasets into a DatasetDict
    codec_dataset = DatasetDict({"train": train_dataset, "test": test_dataset})

    # Push the dataset to the Hugging Face hub, retrying if it fails
    while True:
        try:
            codec_dataset.push_to_hub(args.output_dataset)
            break
        except Exception as e:
            print(e)
