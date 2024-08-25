import argparse
import glob
import json
import os
import re
import subprocess

import tqdm


def lyrics_in_interval(
    lyrics: str, start_interval: int, end_interval: int, exact_match: bool = False
) -> str:
    """Extract lyrics that fall within a specified time interval.

    Args:
        lyrics (str): the lyrics to extract from
        start_interval (int): the start of the time interval
        end_interval (int): the end of the time interval
        exact_match (bool, optional): whether to include only lyrics that exactly match the interval. Defaults to False.

    Returns:
        str: the lyrics that fall within the time interval
    """
    # Use a regular expression to capture time intervals and the text
    pattern = r"\[(\d+\.?\d*):(\d+\.?\d*)(?:-(\d+\.?\d*):(\d+\.?\d*))?\] (.+)"

    # Initialize a list to store lyrics that match the interval
    lyrics_in_interval = []

    # Iterate over each line of the lyrics
    for line in lyrics.splitlines():
        match = re.match(pattern, line)
        if match:
            start1 = float(match.group(1))
            end1 = float(match.group(2))
            start2 = match.group(3)
            end2 = match.group(4)
            text = match.group(5)

            # If a second interval exists, parse them as floats
            intervals = [(start1, end1)]
            if start2 and end2:
                intervals.append((float(start2), float(end2)))

            include_entire_lyric = False

            for start, end in intervals:
                # Determine the overlap between the interval and the lyric time range
                overlap_start = max(start, start_interval)
                overlap_end = min(end, end_interval)

                if overlap_start < overlap_end:  # There is an overlap
                    include_entire_lyric = True
                    break

            if include_entire_lyric:
                # If the lyric overlaps at all, include the entire text
                lyrics_in_interval.append(f"[{start1}:{end1}] {text}")
            elif exact_match:
                # Check if the entire interval fits within the range
                if start_interval <= start1 and end1 <= end_interval:
                    lyrics_in_interval.append(f"[{start1}:{end1}] {text}")
            else:
                # Calculate the portion of text that is within the time interval
                for start, end in intervals:
                    total_time = end - start
                    overlap_start = max(start, start_interval)
                    overlap_end = min(end, end_interval)

                    if overlap_start < overlap_end:  # There's an overlap
                        # Calculate the proportion of the text to include
                        start_index = int(
                            (overlap_start - start) / total_time * len(text)
                        )
                        end_index = int((overlap_end - start) / total_time * len(text))

                        partial_lyric = text[start_index:end_index].strip()
                        lyrics_in_interval.append(
                            f"[{overlap_start}:{overlap_end}] {partial_lyric}"
                        )

    # Return the lyrics as a text, separated by newlines
    return "\n".join(lyrics_in_interval)


args = argparse.ArgumentParser()
args.add_argument("--directory", type=str, required=True)
args.add_argument("--output", type=str, required=True)
args = args.parse_args()

BASE_DIR = args.directory
OUTPUT_DIR = args.output


# Ensure the output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Loop over each MP3 file in the base directory
for audio_file in tqdm.tqdm(glob.glob(BASE_DIR + "*.mp3"), desc="Split audio"):
    base_name = os.path.basename(audio_file).rsplit(".", 1)[0]

    # Check if corresponding JSON metadata file exists
    metadata_file = os.path.join(BASE_DIR, base_name + ".json")
    if not os.path.exists(metadata_file):
        continue

    # Read metadata
    with open(metadata_file, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    # Use ffmpeg to split the audio file into 30-second chunks
    output_pattern = os.path.join(OUTPUT_DIR, base_name + "_%03d.mp3")
    command = [
        "ffmpeg",
        "-i",
        audio_file,
        "-f",
        "segment",
        "-segment_time",
        "30",
        "-c",
        "copy",
        output_pattern,
    ]

    # Run the command without displaying ffmpeg logs
    subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # Process output files and save metadata for each chunk
    for chunk_file in sorted(glob.glob(os.path.join(OUTPUT_DIR, base_name + "_*.mp3"))):
        # Extract start time from the chunk filename
        chunk_base_name = os.path.basename(chunk_file)
        start_time_str = chunk_base_name.split("_")[-1].split(".")[0]
        start_time = int(start_time_str) * 30

        # Update metadata with start time
        chunk_metadata = metadata.copy()
        chunk_metadata["start_time"] = start_time

        # Save updated metadata to JSON file
        with open(
            os.path.join(OUTPUT_DIR, chunk_base_name.rsplit(".", 1)[0] + ".json"),
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(chunk_metadata, f)

# Loop over each MP3 file in the base directory
for file in tqdm.tqdm(glob.glob(OUTPUT_DIR + "*.json"), desc="Split lyrics"):
    with open(file, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    if not metadata["lyrics"]:
        continue

    # Extract the lyrics that fall within the 30-second interval
    lyrics = lyrics_in_interval(
        metadata["lyrics"], metadata["start_time"], metadata["start_time"] + 30
    )

    metadata["lyrics"] = lyrics

    with open(file, "w", encoding="utf-8") as f:
        json.dump(metadata, f)
