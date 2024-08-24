import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

model_name = "gpt2"
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
max_length = model.config.n_positions


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
