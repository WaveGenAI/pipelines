import logging
import os

import numpy as np
import torch
from faster_whisper import WhisperModel

from .utils import evaluate_split_line

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
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = WhisperModel("distil-large-v3", device=device)

    def transcript(self, audio_path: str) -> None | str:
        """Transcribe audio file.

        Args:
            audio_path (str): the path to the audio file

        Returns:
            None | str: the transcribed text
        """

        try:
            # Transcribe audio file
            segments, _ = self.model.transcribe(audio_path, beam_size=5)
        except Exception as e:
            logging.error("Error transcribing %s: %s", audio_path, e)
            return

        probs = []
        lyrics = ""

        for segment in segments:
            probs.append(segment.avg_logprob)

            if (
                np.exp(sum(probs) / len(probs)) < TRANSCRIPT_THRESHOLD
                and len(probs) > 5
            ):
                break

            lyrics += segment.text.strip() + "\n"

        if (
            len(probs) > 0
            and np.exp(sum(probs) / len(probs)) < TRANSCRIPT_THRESHOLD
            and len(probs) > 5
        ):
            logging.warning("Low average logprob: %s", audio_path)
            lyrics = ""

        if evaluate_split_line(lyrics.strip()) < 4:
            logging.warning("Too short: %s", audio_path)
            lyrics = ""

        if lyrics != "":
            logging.info("Lyrics: %s", lyrics.strip())

        return lyrics.strip()
