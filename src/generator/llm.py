"""
LLM model
"""

import logging

import torch
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig


class LLM:
    """
    LLM model that convert clap description in audio prompt for the generation model.
    """

    PROMPT = """ 
    Describe the music with a list of keyword based on information below. Should be the more accurate possible. Don't include timestamp in the description and no-standar character like ':-.'. 
    Write in one unique line. Write nothing about the audio quality (if word noise, ignore it). The information provided may contain errors so try to cross-reference the information as much as possible. Don't describe multiple music.
    The music is called \"{name}\" and the full no-accurate description of the music for each slice of 10 seconds is: {clap}. 
    """

    def __init__(self, llm_name: str = "google/gemma-2-2b-it"):
        quantization_config = BitsAndBytesConfig(load_in_4bit=True)

        self._tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b-it")
        self._model = AutoModelForCausalLM.from_pretrained(
            "google/gemma-2-2b-it",
            quantization_config=quantization_config,
            device_map="auto",
        )

    def generate(self, name: str, clap: str, max_new_tokens: int = 256) -> str:
        """Generate the audio prompt for the LLM model.

        Args:
            name (str): the name of the music.
            clap (str): the clap description of the music.
            max_new_tokens (int, optional): the maximum number of token generated. Defaults to 256.

        Returns:
            str: a string that represent the audio prompt.
        """

        logging.info("Generate prompt for %s", name)
        input_prompt = self.PROMPT.format(clap=clap, name=name).strip()

        input_ids = self._tokenizer(input_prompt, return_tensors="pt").to("cuda")

        outputs = self._model.generate(**input_ids, max_new_tokens=max_new_tokens)

        logging.info("Prompt generated for %s", name)

        return self._tokenizer.decode(outputs[0]).strip()
