import argparse
import os

from datasets import Audio, load_dataset

from pipelines import Downloader

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process datasets to create prompt and download data"
    )
    parser.add_argument(
        "--huggingface", type=str, help="Huggingface dataset name", required=True
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Shuffle the input dataset",
        required=False,
    )
    parser.add_argument(
        "--download", action="store_true", help="Download the dataset", required=False
    )
    args = parser.parse_args()

    dataset = load_dataset(args.huggingface)

    if args.download:
        Downloader(dataset)

    if args.shuffle:
        for split in dataset:
            dataset[split] = dataset[split].shuffle(seed=42)
            dataset[split] = dataset[split].flatten_indices()

    # add audio files to the dataset
    for split in dataset:
        audio_files = []

        for idx, data in enumerate(dataset[split]):
            if os.path.exists(f".pipelines/{split}_{idx}.wav"):
                audio_files.append(os.path.abspath(f".pipelines/{split}_{idx}.wav"))
            else:
                audio_files.append(None)

        dataset[split] = dataset[split].add_column("audio", audio_files)

    # delete all rows without audio
    for split in dataset:
        dataset[split] = dataset[split].filter(lambda x: x["audio"] is not None)

    # cast audio column
    for split in dataset:
        dataset[split] = dataset[split].cast_column("audio", Audio(mono=False))

    print(dataset["train"][0])
