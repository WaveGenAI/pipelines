from datasets import Dataset


class PromptCreator:
    def __init__(self, dataset) -> None:
        self._dataset = dataset

    def create_prompt(self) -> Dataset:
        for split in self._dataset:
            # add column for prompt
            self._dataset[split] = self._dataset[split].add_column(
                "prompt", [None] * len(self._dataset[split])
            )
            for row in self._dataset[split]:
                print(row)
        return self._dataset
