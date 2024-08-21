import logging
import os
from typing import Union

import numpy
import numpy as np
import torch
from faster_whisper import BatchedInferencePipeline, WhisperModel

from .utils import compact_repetitions

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)

TRANSCRIPT_THRESHOLD = 0.6
CMD = """ 
export LD_LIBRARY_PATH=`python3 -c 'import os; import nvidia.cublas.lib; import nvidia.cudnn.lib; import torch; print(os.path.dirname(nvidia.cublas.lib.__file__) + ":" + os.path.dirname(nvidia.cudnn.lib.__file__) + ":" + os.path.dirname(torch.__file__) +"/lib")'`
""".strip()

os.system(CMD)


class TranscriptModel:
    """
    Transcript class using WhisperModel for audio transcription.
    """

    def __init__(self) -> None:
        # get device index
        device_index = []
        for i in range(torch.cuda.device_count()):
            device_index.append(i)

        self.model = WhisperModel(
            "large-v3", compute_type="int8_float16", device_index=device_index
        )
        # self.model = BatchedInferencePipeline(self.model)

    def calculate_logprob(self, lst_prob: list, language_prob: float) -> float:
        """
        Calculate the average logprob of a list of probabilities.
        """

        return np.exp(sum(lst_prob) / (len(lst_prob) + 0.000000001)) * language_prob

    def transcript(self, audio: Union[str, numpy.array]) -> None | str:
        """Transcribe audio file.

        Args:
            audio (Union[str, numpy.array]): The audio file path or numpy array.
        Returns:
            None | str: the transcribed text
        """

        segments, info = self.model.transcribe(audio, beam_size=5)

        language_prob = info.language_probability
        probs = []
        lyrics = ""

        if language_prob >= TRANSCRIPT_THRESHOLD:
            for segment in segments:
                probs.append(segment.avg_logprob)

                if (
                    self.calculate_logprob(probs, language_prob) < TRANSCRIPT_THRESHOLD
                    and len(probs) > 5
                ):
                    break

                lyrics += segment.text.strip() + "\n"

        if self.calculate_logprob(probs, language_prob) < TRANSCRIPT_THRESHOLD:
            logging.warning(
                "Low average logprob: %s", self.calculate_logprob(probs, language_prob)
            )
            lyrics = ""

        lyrics = compact_repetitions(lyrics)

        if len(lyrics) < 150 and len(lyrics) > 0:
            logging.warning("Short lyrics: %s", len(lyrics))
            lyrics = ""

        if lyrics != "":
            logging.info("Lyrics: \n%s", lyrics.strip())
            logging.info(
                "Language probability: %s", self.calculate_logprob(probs, language_prob)
            )

        return lyrics.strip()
