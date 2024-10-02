import argparse

from datasets import load_dataset

from pipelines import Downloader

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process datasets to create prompt and download data"
    )
    parser.add_argument(
        "--huggingface", type=str, help="Huggingface dataset name", required=True
    )
    parser.add_argument(
        "--download", action="store_true", help="Download the dataset", required=False
    )
    args = parser.parse_args()

    dataset = load_dataset(args.huggingface)

    if args.download:
        Downloader(dataset)
