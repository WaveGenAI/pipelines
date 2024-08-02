class Process:
    def iter_over_file(self, input_file: str) -> None:
        """
        Iterate over the file.

        Args:
            input_file (str): The input file.
        """
        with open(input_file, "r", encoding="utf-8") as f:
            for line in f:
                yield line.strip()
