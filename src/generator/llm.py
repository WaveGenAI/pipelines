"""
LLM model
"""

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


class LLM:
    """
    LLM model that convert clap description in audio prompt for the generation model.
    """

    PROMPT = """ 
    Describe the music with a list of keyword based on information below. Should be the more accurate possible. Don't include timestamp in the description and no-standar character like ':-.'. 
    Write in one unique line. Write nothing about the audio quality (if word noise, ignore it) and ban this word: "4 on the floor kick". The information provided may contain errors so try to cross-reference the information as much as possible. Don't describe multiple music.
    Write only tags related to the music split by ",".
    The music is called \"{name}\". The availabe metatags related to the track are {metatags}. The full no-accurate description of the music for each slice of 10 seconds is: {clap}. 
    """

    def __init__(self, llm_name: str = "google/gemma-2-2b-it"):
        quantization_config = BitsAndBytesConfig(load_in_4bit=True)

        self._tokenizer = AutoTokenizer.from_pretrained(llm_name)
        self._model = AutoModelForCausalLM.from_pretrained(
            llm_name,
            quantization_config=quantization_config,
            device_map="auto",
        )

    def generate(self, tags: list, max_new_tokens: int = 256) -> str:
        """Generate the audio prompt for the LLM model.

        Args:
            tags (list): the tags to generate the prompt.
            max_new_tokens (int, optional): the maximum number of token generated. Defaults to 256.

        Returns:
            str: a string that represent the audio prompt.
        """

        prompts_list = []
        for tag in tags:
            name, clap, metatags = tag

            prompt = self.PROMPT.format(clap=clap, name=name, metatags=metatags).strip()

            prompt = self._tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )

            prompts_list.append(prompt)

        input_ids = self._tokenizer(
            prompts_list,
            return_tensors="pt",
            padding=True,
        ).to("cuda")

        outputs_pred = self._model.generate(**input_ids, max_new_tokens=max_new_tokens)
        outputs = []

        for idx, output in enumerate(outputs_pred):
            text = (
                self._tokenizer.decode(output)
                .split("<start_of_turn>model")[1]
                .split("<end_of_turn>")[0]
            ).strip()

            outputs.append({"description": text, "name": tags[idx][0]})

        return outputs
