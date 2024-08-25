import re

import torch
from transformers import pipeline

labels = ["A music recoding with lyrics", "A music recording without lyrics"]
device = "cuda" if torch.cuda.is_available() else "cpu"

audio_classifier = pipeline(
    task="zero-shot-audio-classification",
    model="laion/larger_clap_music_and_speech",
    device=device,
)


def remove_brackets(text: str) -> str:
    """Remove text between brackets [].

    Args:
        text (str): The text to remove the brackets from.

    Returns:
        str: The text with the brackets removed.
    """
    return re.sub(r"\[.*?\]", "", text)


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

        # Extraire le texte sans timestamp pour la comparaison
        match = re.match(r"\[.*?\](.*)", current_line)
        if not match:
            i += 1
            continue
        current_content = match.group(1).strip()
        count = 1
        start_time = re.match(r"\[(.*?)\]", current_line).group(1)

        # Compter les répétitions de la ligne courante
        while (
            i + 1 < len(lines)
            and re.match(r"\[.*?\](.*)", lines[i + 1]).group(1).strip()
            == current_content
        ):
            count += 1
            i += 1

        if count > 1:
            end_time = re.match(r"\[(.*?)\]", lines[i]).group(1)
            compacted_lines.append(
                f"[{start_time}-{end_time}] {current_content} (x{count})"
            )
        else:
            compacted_lines.append(current_line)

        i += 1

    return "\n".join(compacted_lines)


def check_lyrics_repetition(text, threshold=0.2):
    text = remove_brackets(text)
    text = text.replace("\n", "").replace(" ", "")
    total_length = len(text)

    char_count = {}

    for char in text:
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
    text = remove_brackets(text)
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
