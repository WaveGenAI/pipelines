import hashlib
import os
from typing import Any, Dict, List, Union

import torch
from accelerate import Accelerator
from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

from pipelines.utils import hash_url

PROMPT = """ 
You will be given informations related to an audio sample. These informations include:
1. The title
3. The description of the audio sample
4. Some tags related to the audio sample

Your task is to generate a short prompt for a music generation model that describes the audio sample without including any names and titles.
The summary should be concise and capture the essence of the audio sample **without including any names and titles** and start with the word "Prompt:".

Here are some examples of valid prompts:
Prompt: A jazz pop top charts song with emotional vocals, catchy chorus, and trumpet solos.
Prompt: Smooth Contemporary R&B with subtle Electronic elements, featuring a pulsing 104 BPM drum machine beat, filtered synths, lush electric piano, and soaring strings, with an intimate mood.
Prompt: Indie Rock with 90s influences, featuring a combination of clean and distorted guitars, driving drum beats, and a prominent bassline, with a moderate tempo around 120 BPM, and a mix of introspective and uplifting moods, evoking a sense of nostalgia and hope.

Here is the information:
"""


class PromptCreator:
    def __init__(self, dataset, use_cache: bool = True):
        self._dataset = dataset
        self._use_cache = use_cache

        self.accelerator = Accelerator()

        model = AutoModelForCausalLM.from_pretrained(
            "microsoft/Phi-3-mini-4k-instruct",
            device_map="cuda",
            trust_remote_code=True,
            torch_dtype=torch.float16,
        )

        self.model = self.accelerator.prepare(model)
        self.tokenizer = AutoTokenizer.from_pretrained(
            "microsoft/Phi-3-mini-4k-instruct"
        )

    def _data_collector(
        self, features: List[Dict[str, Union[List[int], torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        inputs = self.tokenizer(
            [self.generate_prompt(feature) for feature in features],
            padding="longest",
            return_tensors="pt",
        )
        return inputs

    def generate_prompt(self, row: Dict[str, Any]) -> str:
        """Function to generate prompt for the model

        Args:
            row (Dict[str, Any]): Row of the dataset

        Returns:
            str: Prompt for the model
        """
        informations = ""
        for key, value in row.items():
            if isinstance(value, str):
                informations += f"{key}: {value}\n"[
                    :2000
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

    def generate_step(self, batch: Dict[str, torch.Tensor], url: str) -> str:
        """Function to generate text from the model

        Args:
            batch (Dict[str, torch.Tensor]): Batch of input data
            url (str): URL of the audio file
        Returns:
            torch.Tensor: Output ids from the model
        """

        # check if the prompt is already generated
        file_name = hash_url(url)
        if self._use_cache and os.path.exists(f".pipelines/{file_name}.txt"):
            with open(f".pipelines/{file_name}.txt", "r") as file:
                return file.read()

        output_ids = self.model.generate(
            batch["input_ids"],
            attention_mask=batch["attention_mask"],
            max_new_tokens=500,
        )

        prompt = self.tokenizer.decode(output_ids[0], skip_special_tokens=True).rsplit(
            "Prompt:", 1
        )[1]

        with open(f".pipelines/{file_name}.txt", "w") as file:
            file.write(prompt)

        return prompt

    def create_prompt(self) -> Dataset:
        for split in self._dataset:
            data_loader = DataLoader(
                self._dataset[split],
                batch_size=1,
                num_workers=4,
                pin_memory=True,
                collate_fn=self._data_collector,
            )

            data_loader = self.accelerator.prepare(data_loader)

            generated_prompts = []
            for idx, batch in enumerate(data_loader):
                prompt = self.generate_step(batch, self._dataset[split][idx]["url"])
                print(prompt)
                print(self._dataset[split][idx]["url"])
                generated_prompts.append(prompt)

            # add a column for the generated prompts
            self._dataset[split] = self._dataset[split].add_column(
                "generated_prompt", generated_prompts
            )

        return self._dataset
