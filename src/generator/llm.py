"""
LLM model
"""

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


class LLM:
    """
    LLM model that convert clap description in audio prompt for the generation model.
    """

    PROMPT = """ 
    Describe the music by generating a list of keywords/tags based on information below. Deduct tag with the name of the music, the name of the artist, the genre, the key features of the musics and the bad quality description of the music for each slice of 10 seconds.
    Be the more accurate as possible.
    
    The music is called \"{name}\". 
    The availabe metatags related to the track are: 
    {metatags}. 
    
    The full no-accurate description of the music for each slice of 10 seconds is: 
    {clap}. 
    
    The features of the music are: 
    {features}.
    
    Write nothing except the tags. Don't include timestamp from the no-accurate description and ignore tags about the audio quality.
    e.g: Italo disco,  Italo-disco,  1980s, 124BPM, '80s,  virtuoso synthwave,  euro dance, Key F, Chords: Fm, Cm, Bb, Gm
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
            name, clap, metatags, features = tag

            prompt = self.PROMPT.format(
                clap=clap[:1000], name=name, metatags=metatags, features=features
            ).strip()
            print(prompt)
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
