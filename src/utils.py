"""
Utility functions.
"""


def save_prompt(file_path: str, prompt: str) -> None:
    """Save the prompt to a file.

    Args:
        file_path (str): the file path.
        prompt (str): the prompt.
    """
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(prompt)
