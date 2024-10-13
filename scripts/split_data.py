import argparse
import glob
import os
import subprocess

import tqdm

args = argparse.ArgumentParser()
args.add_argument("--input_dir", type=str, required=True)
args.add_argument("--chunk-size", type=int, default=30)
args.add_argument("--output_dir", type=str, required=True)
args.add_argument("--remove-original", action="store_true")
args = args.parse_args()

for audio_file in tqdm.tqdm(glob.glob(args.input_dir + "*.mp3"), desc="Split audio"):
    base_name = os.path.basename(audio_file).rsplit(".", 1)[0]

    # Check if corresponding JSON metadata file exists
    metadata_file = os.path.join(args.input_dir, base_name + ".txt")
    if not os.path.exists(metadata_file):
        continue

    # Use ffmpeg to split the audio file into CHUNK_SIZE-second chunks
    output_pattern = os.path.join(args.output_dir, base_name + "_%03d.mp3")
    command = [
        "ffmpeg",
        "-i",
        audio_file,
        "-f",
        "segment",
        "-segment_time",
        str(args.chunk_size),
        "-c",
        "copy",
        output_pattern,
    ]

    # Run the command without displaying ffmpeg logs
    subprocess.run(
        command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True
    )

    # copy metadata file to output directory
    if not os.path.exists(os.path.join(args.output_dir, base_name + ".txt")):
        metadata_output = os.path.join(args.output_dir, base_name + ".txt")
        subprocess.run(["cp", metadata_file, metadata_output], check=True)

    # Remove the original audio file
    if args.remove_original:
        os.remove(audio_file)
