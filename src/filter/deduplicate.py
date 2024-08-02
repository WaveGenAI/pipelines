"""
Deduplicate filter module.
"""

import logging

from src.filter.filter import Filter


class DeduplicateFilter(Filter):
    """
    Deduplicate filter class.
    """

    def filter(self, filename: str) -> None:
        """
        Deduplicate the data in the file.

        Args:
            filename (str): The filename to filter.
        """

        with open(filename, "r", encoding="utf-8") as file:
            lines = file.readlines()

        nbm_lines = len(lines)
        lines = list(set(lines))

        logging.info("Deduplicate filter: %s -> %s", nbm_lines, len(lines))

        with open(filename, "w", encoding="utf-8") as file:
            file.writelines(lines)

        logging.info("Deduplicate filter done.")

    def type(self) -> str:
        return "text"
