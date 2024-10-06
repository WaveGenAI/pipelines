import hashlib

import librosa
import numpy as np


def hash_url(url: str) -> str:
    """Function to hash the url

    Args:
        url (str): URL to hash

    Returns:
        str: Hashed URL
    """
    return hashlib.sha256(url.encode("utf-8")).hexdigest()


def get_bpm(audio: dict, duration: int = 30) -> int:
    """Function to get the BPM of the audio

    Args:
        audio (dict): Audio dictionary containing the array and sampling rate
        duration (int, optional): Duration of the audio to consider

    Returns:
        int: BPM of the audio
    """
    sampling_rate = audio["sampling_rate"]
    onset_env = librosa.onset.onset_strength(
        y=audio["array"][..., : sampling_rate * duration], sr=sampling_rate
    )
    tempo = librosa.beat.tempo(onset_envelope=onset_env, sr=sampling_rate)

    return int(tempo[0][0])
