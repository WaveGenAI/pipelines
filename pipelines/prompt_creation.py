import logging
import os
from typing import Any, Dict, List, Union

import torch
from accelerate import Accelerator
from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from pipelines.utils import get_bpm, hash_url

PROMPT = """ 
You will be given informations related to an audio sample. These informations include:
1. The title
3. The description of the audio sample
4. The tempo of the audio sample
5. Some tags related to the audio sample

Your task is to generate a short prompt for a music generation model that describes the audio sample without including any names and titles.
The summary should be concise and capture the essence of the audio sample **without including any names and titles** and start with the word "Prompt:".

Here are some examples of valid prompts:
Prompt: A jazz pop top charts song with emotional vocals, catchy chorus, and trumpet solos.
Prompt: Smooth Contemporary R&B with subtle Electronic elements, featuring a pulsing 104 BPM drum machine beat, filtered synths, lush electric piano, and soaring strings, with an intimate mood.
Prompt: Indie Rock with 90s influences, featuring a combination of clean and distorted guitars, driving drum beats, and a prominent bassline, with a moderate tempo around 120 BPM, and a mix of introspective and uplifting moods, evoking a sense of nostalgia and hope.

Here is the information:
"""


def get_current_device() -> int:
    """Get the current device. For GPU we return the local process index to enable multiple GPU training."""
    return Accelerator().local_process_index if torch.cuda.is_available() else "cpu"


def get_kbit_device_map() -> Union[Dict[str, int], None]:
    """Useful for running inference with quantized models by setting `device_map=get_peft_device_map()`"""
    return {"": get_current_device()} if torch.cuda.is_available() else None


class PromptCreator:
    """Class to generate prompts"""

    def __init__(
        self,
        dataset,
        use_cache: bool = True,
        batch_size: int = 1,
        cache_dir: str = ".pipelines",
    ):
        self._dataset = dataset
        self._use_cache = use_cache
        self._batch_size = batch_size
        self._cache_dir = cache_dir

        self.accelerator = Accelerator()

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16
        )

        model = AutoModelForCausalLM.from_pretrained(
            "microsoft/Phi-3-mini-4k-instruct",
            trust_remote_code=True,
            quantization_config=quantization_config,
            device_map=get_kbit_device_map(),
            low_cpu_mem_usage=True,
        )

        self.model = self.accelerator.prepare(model)
        self.tokenizer = AutoTokenizer.from_pretrained(
            "microsoft/Phi-3-mini-4k-instruct"
        )

        self._logger = logging.getLogger(__name__)

    def _data_collector(
        self, features: List[Dict[str, Union[List[int], torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        inputs = self.tokenizer(
            [self._generate_prompt(feature) for feature in features],
            padding="longest",
            return_tensors="pt",
        )
        return inputs

    def _generate_prompt(self, row: Dict[str, Any]) -> str:
        """Function to generate prompt for the model

        Args:
            row (Dict[str, Any]): Row of the dataset

        Returns:
            str: Prompt for the model
        """

        bpm = get_bpm(row["audio"])
        row["BPM"] = bpm

        informations = ""
        for key, value in row.items():
            if isinstance(value, str):
                informations += f"{key}: {value}\n"[
                    :1000
                ]  # limit the length of the prompt for vram

        chat = [
            {
                "role": "system",
                "content": "You are a robot that describes audio samples for music generation models that dont write names and titles for copyright reasons.",
            },
            {"role": "user", "content": PROMPT + informations + "\nPrompt: "},
        ]

        chat = self.tokenizer.apply_chat_template(
            chat, tokenize=False, add_generation_prompt=True
        )

        return chat

    def _load_prompt(self, sample: Dict[str, Any]) -> str:
        """Load prompt from cache if it exists, otherwise return an empty string"""

        file_name = hash_url(sample["url"])
        cache_file = os.path.join(self._cache_dir, f"{file_name}.txt")

        if os.path.exists(cache_file):
            with open(cache_file, "r", encoding="utf-8") as file:
                return file.read()

        return ""

    def _generate_step(self, batch: Dict[str, torch.Tensor]) -> List[str]:
        """Function to generate text from the model

        Args:
            batch (Dict[str, torch.Tensor]): Batch of input data
        Returns:
            torch.Tensor: Output ids from the model
        """

        output_ids = self.model.generate(
            batch["input_ids"],
            attention_mask=batch["attention_mask"],
            max_new_tokens=500,
        )

        generate_ids = output_ids[:, batch["input_ids"].shape[1] :]
        prompt = self.tokenizer.batch_decode(generate_ids, skip_special_tokens=True)

        # remove the first prompt word
        prompt = [p.replace("Prompt: ", "", 1).strip() for p in prompt]

        return prompt

    def _filter_dataset(self, dataset: Dataset) -> Dataset:
        """Filter the dataset to keep only samples without generated prompts."""

        def needs_generation(sample):
            file_name = hash_url(sample["url"])
            cache_file = os.path.join(self._cache_dir, f"{file_name}.txt")
            return not os.path.exists(cache_file)

        return dataset.filter(needs_generation)

    def create_prompt(self) -> Dataset:
        """Start the prompt creation process

        Returns:
            Dataset: Dataset with the generated prompts
        """
        for split in self._dataset:
            filtered_dataset = self._dataset[split]
            if self._use_cache:
                filtered_dataset = self._filter_dataset(self._dataset[split])

            data_loader = DataLoader(
                filtered_dataset,
                batch_size=self._batch_size,
                num_workers=self._batch_size + 3,
                pin_memory=True,
                collate_fn=self._data_collector,
            )

            data_loader = self.accelerator.prepare(data_loader)

            # Generate prompts for filtered samples
            for idx, batch in enumerate(data_loader):
                prompts = self._generate_step(batch)
                for idx_2, prompt in enumerate(prompts):
                    url = filtered_dataset[idx * self._batch_size + idx_2]["url"]
                    file_name = hash_url(url)
                    with open(
                        os.path.join(self._cache_dir, f"{file_name}.txt"),
                        "w",
                        encoding="utf-8",
                    ) as file:
                        file.write(prompt)

                    self._logger.info("Generated prompt for %s: \n%s", url, prompt)

            self.accelerator.wait_for_everyone()

            all_prompts = [self._load_prompt(sample) for sample in self._dataset[split]]
            # add a column for the generated prompts
            self._dataset[split] = self._dataset[split].add_column(
                "generated_prompt", all_prompts
            )

        return self._dataset
