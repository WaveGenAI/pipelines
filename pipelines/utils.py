import hashlib

import librosa
import soundfile as sf


def hash_url(url: str) -> str:
    """Function to hash the url

    Args:
        url (str): URL to hash

    Returns:
        str: Hashed URL
    """
    return hashlib.sha256(url.encode("utf-8")).hexdigest()


def get_bpm(audio: str, duration: int = 30) -> int:
    """Function to get the BPM of the audio

    Args:
        audio (str): Audio file path
        duration (int, optional): Duration of the audio to consider

    Returns:
        int: BPM of the audio
    """
    # get sampling rate of the audio without loading the whole file (for ram efficiency)
    sampling_rate = sf.info(audio).samplerate

    audio, sampling_rate = sf.read(audio, frames=sampling_rate * duration)
    onset_env = librosa.onset.onset_strength(y=audio.T, sr=sampling_rate)
    tempo = librosa.beat.tempo(onset_envelope=onset_env, sr=sampling_rate)[0]

    return int(tempo)
