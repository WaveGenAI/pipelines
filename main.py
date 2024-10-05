import argparse
import hashlib
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Union

import torch
from accelerate import Accelerator
from datasets import Audio, load_dataset
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

from pipelines import Downloader, PromptCreator


def generate_prompt(row: Dict[str, Any]) -> str:
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
        {"role": "user", "content": PROMPT + informations},
    ]

    chat = tokenizer.apply_chat_template(
        chat, tokenize=False, add_generation_prompt=True
    )

    return chat


def hash_url(url: str) -> str:
    """Function to hash the url

    Args:
        url (str): URL to hash

    Returns:
        str: Hashed URL
    """
    return hashlib.sha256(url.encode("utf-8")).hexdigest()


@dataclass
class DataCollatorWithPadding:
    """
    Data collator that will dynamically pad the inputs received to the longest sequence in the batch.
    """

    tokenizer: Any

    def __call__(
        self, features: List[Dict[str, Union[List[int], torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        inputs = self.tokenizer(
            [generate_prompt(feature) for feature in features],
            padding="longest",
            return_tensors="pt",
        )
        return inputs


def generate_step(batch: Dict[str, torch.Tensor], url: str) -> str:
    """Function to generate text from the model

    Args:
        batch (Dict[str, torch.Tensor]): Batch of input data
        url (str): URL of the audio file
    Returns:
        torch.Tensor: Output ids from the model
    """

    # check if the prompt is already generated
    file_name = hash_url(url)
    if os.path.exists(f".pipelines/{file_name}.txt"):
        with open(f".pipelines/{file_name}.txt", "r") as file:
            return file.read()

    output_ids = model.generate(
        batch["input_ids"],
        attention_mask=batch["attention_mask"],
        max_new_tokens=500,
    )

    prompt = tokenizer.decode(output_ids[0], skip_special_tokens=True).rsplit(
        "Prompt:", 1
    )[1]

    with open(f".pipelines/{file_name}.txt", "w") as file:
        file.write(prompt)

    return prompt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process datasets to create prompt and download data"
    )
    parser.add_argument(
        "--huggingface", type=str, help="Huggingface dataset name", required=True
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Shuffle the input dataset",
        required=False,
    )
    parser.add_argument(
        "--download", action="store_true", help="Download the dataset", required=False
    )

    args = parser.parse_args()

    dataset = load_dataset(args.huggingface)

    if args.shuffle:
        for split in dataset:
            dataset[split] = dataset[split].shuffle(seed=42)
            dataset[split] = dataset[split].flatten_indices()

    if args.download:
        Downloader(dataset)

    accelerator = Accelerator()

    model = AutoModelForCausalLM.from_pretrained(
        "microsoft/Phi-3-mini-4k-instruct",
        device_map="cuda",
        trust_remote_code=True,
        torch_dtype=torch.float16,
    )

    model = accelerator.prepare(model)
    tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct")

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

    data_collator = DataCollatorWithPadding(tokenizer)

    # add audio files to the dataset
    for split in dataset:
        audio_files = []

        for data in dataset[split]:
            file_name = hash_url(data["url"])
            if os.path.exists(f".pipelines/{file_name}.wav"):
                audio_files.append(os.path.abspath(f".pipelines/{file_name}.wav"))
            else:
                audio_files.append(None)

        dataset[split] = dataset[split].add_column("audio", audio_files)

    # delete all rows without audio
    for split in dataset:
        dataset[split] = dataset[split].filter(lambda x: x["audio"] is not None)

    # cast audio column
    for split in dataset:
        dataset[split] = dataset[split].cast_column("audio", Audio(mono=False))

    for split in dataset:
        data_loader = DataLoader(
            dataset[split],
            batch_size=1,
            num_workers=4,
            pin_memory=True,
            collate_fn=data_collator,
        )

        data_loader = accelerator.prepare(data_loader)

        generated_prompts = []
        for idx, batch in enumerate(data_loader):
            prompt = generate_step(batch, dataset[split][idx]["url"])
            print(prompt)
            print(dataset[split][idx]["url"])
            generated_prompts.append(prompt)

        # add a column for the generated prompts
        dataset[split] = dataset[split].add_column(
            "generated_prompt", generated_prompts
        )
