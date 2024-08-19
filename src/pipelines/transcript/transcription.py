import logging
import os

import numpy as np
import torch
from faster_whisper import WhisperModel

from .utils import compact_repetitions

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)

TRANSCRIPT_THRESHOLD = 0.4
CMD = """ 
export LD_LIBRARY_PATH=`python3 -c 'import os; import nvidia.cublas.lib; import nvidia.cudnn.lib; import torch; print(os.path.dirname(nvidia.cublas.lib.__file__) + ":" + os.path.dirname(nvidia.cudnn.lib.__file__) + ":" + os.path.dirname(torch.__file__) +"/lib")'`
""".strip()

os.system(CMD)


class TranscriptModel:
    """
    Transcript class using WhisperModel for audio transcription.
    """

    def __init__(self) -> None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = WhisperModel("large-v3", device=device)

    def calculate_logprob(self, lst_prob: list, language_prob: float) -> float:
        """
        Calculate the average logprob of a list of probabilities.
        """

        return np.exp(sum(lst_prob) / (len(lst_prob) + 0.000000001)) * language_prob

    def transcript(self, audio_path: str) -> None | str:
        """Transcribe audio file.

        Args:
            audio_path (str): the path to the audio file

        Returns:
            None | str: the transcribed text
        """

        try:
            # Transcribe audio file
            segments, info = self.model.transcribe(audio_path, beam_size=5)
        except Exception as e:
            logging.error("Error transcribing %s: %s", audio_path, e)
            return

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
            logging.warning("Low average logprob: %s", audio_path)
            lyrics = ""

        if len(lyrics) < 150 and len(lyrics) > 0:
            logging.warning("Short lyrics: %s", audio_path)
            lyrics = ""

        lyrics = compact_repetitions(lyrics)

        if lyrics != "":
            logging.info("Lyrics: %s", lyrics.strip())
            logging.info(
                "Language probability: %s", self.calculate_logprob(probs, language_prob)
            )

        return lyrics.strip()
