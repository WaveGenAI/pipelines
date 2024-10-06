import argparse
import os

from datasets import Audio, load_dataset

from pipelines import Downloader, PromptCreator
from pipelines.utils import get_bpm, hash_url

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
    parser.add_argument(
        "--use_cache",
        action="store_true",
        help="Use cache for prompt generation",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for prompt generation",
    )
    parser.add_argument(
        "--output_dataset",
        type=str,
        default=None,
        help="Output dataset name",
        required=True,
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
    args = parser.parse_args()

    dataset = load_dataset(args.huggingface)

    if args.shuffle:
        for split in dataset:
            dataset[split] = dataset[split].shuffle(seed=42)
            dataset[split] = dataset[split].flatten_indices()

    if args.download:
        Downloader(dataset, cache_dir=args.cache_dir, max_files=args.max_files)

    # add audio files to the dataset
    for split in dataset:
        audio_files = []

        for data in dataset[split]:
            file_name = hash_url(data["url"])
            if os.path.exists(os.path.join(args.cache_dir, f"{file_name}.mp3")):
                audio_files.append(os.path.join(args.cache_dir, f"{file_name}.mp3"))
            else:
                audio_files.append(None)

        dataset[split] = dataset[split].add_column("audio", audio_files)

    # delete all rows without audio
    for split in dataset:
        dataset[split] = dataset[split].filter(lambda x: x["audio"] is not None)

    # cast audio column
    for split in dataset:
        dataset[split] = dataset[split].cast_column(
            "audio", Audio(mono=False, sampling_rate=44100)
        )

    dataset = PromptCreator(
        dataset,
        use_cache=args.use_cache,
        batch_size=args.batch_size,
        cache_dir=args.cache_dir,
    ).create_prompt()

    dataset.push_to_hub(args.output_dataset)
