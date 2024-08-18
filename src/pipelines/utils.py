def evaluate_split_line(text: str) -> int:
    text = text.strip().split("\n")

    avg_nbm = []
    for line in text:
        avg_nbm.append(len(line.split(" ")))

    return sum(avg_nbm) // len(avg_nbm)
