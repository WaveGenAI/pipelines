import logging
import os
import subprocess
from typing import Union

import numpy
import numpy as np
import torch
from faster_whisper import WhisperModel
from transformers import pipeline
from transformers.utils import is_flash_attn_2_available

from .utils import compact_repetitions

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)

# RUN export LD_LIBRARY_PATH=`python3 -c 'import os; import nvidia.cublas.lib; import nvidia.cudnn.lib; import torch; print(os.path.dirname(nvidia.cublas.lib.__file__) + ":" + os.path.dirname(nvidia.cudnn.lib.__file__) + ":" + os.path.dirname(torch.__file__) +"/lib")'`
# before running the script

TRANSCRIPT_THRESHOLD = 0.6


class TranscriptModel:
    """
    Transcript class using WhisperModel for audio transcription.
    """

    def __init__(self) -> None:
        # get device index
        device_index = []
        for i in range(torch.cuda.device_count()):
            device_index.append(i)

        self.validation_model = WhisperModel(
            "large-v3", compute_type="int8_float16", device_index=device_index
        )

        self.transcription_model = pipeline(
            "automatic-speech-recognition",
            model="openai/whisper-large-v3",
            torch_dtype=torch.float16,
            device="cuda",
            model_kwargs=(
                {"attn_implementation": "flash_attention_2"}
                if is_flash_attn_2_available()
                else {"attn_implementation": "sdpa"}
            ),
        )

    def calculate_logprob(self, lst_prob: list, language_prob: float) -> float:
        """
        Calculate the average logprob of a list of probabilities.
        """

        return np.exp(sum(lst_prob) / (len(lst_prob) + 0.000000001)) * language_prob

    def contain_lyrics(self, audio: Union[str, numpy.array]) -> bool:
        """
        Check if the audio file contains lyrics.
        """

        segments, info = self.validation_model.transcribe(audio, beam_size=5)
        language_prob = info.language_probability
        probs = []

        # true if the language detected is higger than the threshold
        is_lyrics = info.language_probability >= TRANSCRIPT_THRESHOLD

        if language_prob >= TRANSCRIPT_THRESHOLD:
            for segment in segments:
                probs.append(segment.avg_logprob)

                if (
                    self.calculate_logprob(probs, language_prob) < TRANSCRIPT_THRESHOLD
                    and len(probs) > 5
                ):
                    is_lyrics = False
                    break

                if len(probs) > 5:
                    break

        return is_lyrics

    def transcript(
        self, audio: Union[str, numpy.array], check_lyrics: bool = True
    ) -> None | str:
        """Transcribe audio file.

        Args:
            audio (Union[str, numpy.array]): The audio file path or numpy array.
            check_lyrics (bool): Whether to check if the audio contains lyrics.
        Returns:
            None | str: the transcribed text
        """

        if check_lyrics and not self.contain_lyrics(audio):
            return ""

        # use insanely-fast-whisper for transcription (faster but don't return confidence)
        lyrics = self.transcription_model(
            audio,
            chunk_length_s=30,
            batch_size=5,
            return_timestamps=True,
        )["text"]

        lyrics = compact_repetitions(lyrics)

        if len(lyrics) < 150 and len(lyrics) > 0:
            logging.warning("Short lyrics: %s", len(lyrics))
            lyrics = ""

        return lyrics.strip()
