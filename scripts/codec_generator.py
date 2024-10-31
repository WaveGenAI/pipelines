import argparse
import glob
import os
import random

import dac
import torch
import tqdm
import webdataset as wds
from audiotools import AudioSignal

from pipelines.audio_mae import AudioMAEConfig, PretrainedAudioMAEEncoder


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dataset",
        type=str,
        required=True,
        help="Directory containing the audio files",
    )
    parser.add_argument(
        "--output_dataset",
        type=str,
        required=True,
        help="Output directory for the WebDataset",
    )
    parser.add_argument(
        "--test_size",
        type=int,
        default=2000,
        help="Number of files to use for the test set",
    )
    parser.add_argument(
        "--max_files", type=int, default=None, help="Maximum number of files to process"
    )
    return parser.parse_args()


def setup_models(device):
    model_path = dac.utils.download(model_type="44khz")
    model = dac.DAC.load(model_path).to(device)

    config = AudioMAEConfig()
    mae = PretrainedAudioMAEEncoder(config).load_model().to(device)

    return model, mae


def process_audio_file(audio_file, model, mae):
    """Process an audio file to extract codes and embeddings"""
    try:
        audio = AudioSignal(audio_file)
    except Exception as e:
        return None

    # resample the audio to 44.1kHz
    audio = audio.resample(44100)

    base_name = audio_file.split("_")[0]

    #  get the prompt
    prompt_file = base_name + ".txt"
    if not os.path.exists(prompt_file):
        return None

    with open(prompt_file, "r", encoding="utf-8") as f:
        prompt = f.read().strip()

    audio_data = audio.audio_data.permute(1, 0, 2)

    with torch.no_grad():
        embd = mae.forward(audio_file)
        x = model.preprocess(audio_data.to(model.device), audio.sample_rate)
        _, codes, _, _, _ = model.encode(x)

    return {
        "codes": codes.long().cpu().numpy(),
        "embd": embd.cpu().numpy(),
        "prompt": prompt,
    }


def create_webdataset(args):
    """Create a WebDataset from audio files"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, mae = setup_models(device)

    # create output directory
    os.makedirs(args.output_dataset, exist_ok=True)
    os.makedirs(os.path.join(args.output_dataset, "data"), exist_ok=True)

    # Parcours des fichiers audio
    audio_files = glob.glob(os.path.join(args.input_dataset, "*.mp3"))

    # Split audio files
    train_files = audio_files[args.test_size :]
    test_files = audio_files[: args.test_size]

    # shuffle the files
    random.shuffle(train_files)
    random.shuffle(test_files)

    def process_dataset(files, split):
        # limit to 1GB per shard
        sink = wds.ShardWriter(
            os.path.join(args.output_dataset, f"data/{split}-%06d.tar"), maxsize=1e9
        )

        for idx, audio_file in enumerate(tqdm.tqdm(files, total=args.max_files)):
            if args.max_files is not None and idx >= args.max_files:
                break

            result = process_audio_file(audio_file, model, mae)

            if result is not None:
                sample = {
                    "__key__": f"sample_{idx}",
                    "codes.npy": result["codes"],
                    "embd.npy": result["embd"],
                    "prompt.txt": result["prompt"],
                }
                sink.write(sample)

        sink.close()

    # process_dataset(train_files, "train")
    process_dataset(train_files, "train")
    process_dataset(test_files, "test")

    print(f"WebDataset created in {args.output_dataset}")


def main():
    args = parse_arguments()
    create_webdataset(args)


if __name__ == "__main__":
    main()
