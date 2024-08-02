"""
Space filter module.
"""

import logging

from src.filter.filter import Filter


class SpaceFilter(Filter):
    """
    Space filter class.
    """

    def filter(self, filename: str) -> None:
        """
        Space the data in the file.

        Args:
            filename (str): The filename to filter.
        """

        with open(filename, "r", encoding="utf-8") as file:
            text = file.read()

        # if there are , without space, add space
        text = text.replace(",", ", ").replace("  ", " ")

        with open(filename, "w", encoding="utf-8") as file:
            file.write(text)

    def type(self) -> str:
        return "line"
