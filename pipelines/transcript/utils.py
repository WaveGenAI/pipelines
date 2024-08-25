import torch
from transformers import pipeline

labels = ["A music recoding with lyrics", "A music recording without lyrics"]
device = "cuda" if torch.cuda.is_available() else "cpu"

audio_classifier = pipeline(
    task="zero-shot-audio-classification",
    model="laion/larger_clap_music_and_speech",
    device=device,
)


def compact_repetitions(text: str) -> str:
    """Compact repeated lines in a text.

    Args:
        text (str): The text to compact.

    Returns:
        str: The compacted text.
    """

    lines = text.strip().splitlines()
    compacted_lines = []
    i = 0

    while i < len(lines):
        current_line = lines[i]
        count = 1

        # Count how many times the current line repeats
        while (
            i + 1 < len(lines) and lines[i + 1] == current_line and current_line.strip()
        ):
            count += 1
            i += 1

        if count > 1 and current_line.strip():
            compacted_lines.append(f"{current_line} (x{count})")
        else:
            compacted_lines.append(current_line)

        i += 1

    return "\n".join(compacted_lines)


def check_lyrics_repetition(lyrics, threshold=0.2):
    lyrics = lyrics.replace("\n", "").replace(" ", "")
    total_length = len(lyrics)

    char_count = {}

    for char in lyrics:
        if char in char_count:
            char_count[char] += 1
        else:
            char_count[char] = 1

    for char, count in char_count.items():
        if count > threshold * total_length:
            return False

    return True


def get_lines_ratio(text: str) -> float:
    """Find the ratio of characters to lines in a text.

    Args:
        text (str): The text to analyze.

    Returns:
        float: The ratio of characters to lines.
    """
    num_char = len(text)
    num_line = len(text.splitlines())

    return num_char / (num_line + 0.0001)


def is_contain_lyrics_clap(audio_path: str) -> tuple[bool, float]:
    """Function to check if the audio contains lyrics.

    Args:
        audio_path (str): The audio file path.

    Returns:
        tuple[bool, float]: Whether the audio contains lyrics and the difference between the highest and lowest score.
    """
    results = audio_classifier(
        audio_path,
        candidate_labels=labels,
    )

    top_label = results[0]["label"]
    results = [pred["score"] for pred in results]

    return (
        top_label == "A music recoding with lyrics",
        max(results) - min(results),
    )
