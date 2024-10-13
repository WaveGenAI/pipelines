import argparse

from datasets import load_dataset

from pipelines.download import Downloader

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process datasets to create prompt and download data"
    )
    parser.add_argument(
        "--input_dataset",
        type=str,
        help="Huggingface dataset name",
        required=True,
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Shuffle the input dataset",
        required=False,
    )

    parser.add_argument(
        "--cache_dir",
        type=str,
        default=".pipelines",
        help="Cache directory",
        required=False,
    )
    parser.add_argument(
        "--max_files",
        type=int,
        default=0,
        help="Maximum number of files to download",
        required=False,
    )
    parser.add_argument(
        "--audio_duration",
        type=int,
        default=60 * 10,
        help="Duration of the audio file",
        required=False,
    )
    args = parser.parse_args()

    dataset = load_dataset(args.input_dataset)

    if args.shuffle:
        for split in dataset:
            dataset[split] = dataset[split].shuffle(seed=42)
            dataset[split] = dataset[split].flatten_indices()  # for performance

    Downloader(
        dataset,
        cache_dir=args.cache_dir,
        max_files=args.max_files,
        audio_duration=args.audio_duration,
    )
