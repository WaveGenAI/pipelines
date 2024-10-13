import argparse
import os

from datasets import load_dataset

from pipelines import PromptCreator
from pipelines.utils import hash_url

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
        "--cache_dir",
        type=str,
        default=".pipelines",
        help="Cache directory",
        required=False,
    )

    args = parser.parse_args()

    dataset = load_dataset(args.input_dataset)

    # add audio files to the dataset
    for split in dataset:
        audio_files = []

        # get existing audio files
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

    dataset = PromptCreator(
        dataset,
        use_cache=args.use_cache,
        batch_size=args.batch_size,
        cache_dir=args.cache_dir,
    ).create_prompt()
