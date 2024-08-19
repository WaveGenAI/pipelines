def evaluate_split_line(text: str) -> int:
    text = text.strip().split("\n")

    avg_nbm = []
    for line in text:
        avg_nbm.append(len(line.split(" ")))

    return sum(avg_nbm) // len(avg_nbm)


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
