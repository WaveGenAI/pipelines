import hashlib

import ffmpeg
import librosa
import soundfile as sf
import os


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
    tempo = librosa.beat.tempo(onset_envelope=onset_env, sr=sampling_rate)

    return int(tempo[0][0])


def cut_audio(file_path: str, duration: int):
    """Function to cut the audio file to the given duration

    Args:
        file_path (str): Path to the audio file
        duration (int): Duration to cut the audio file
    """
    # copy the input file to avoid overwriting the original file
    ffmpeg.input(file_path).output(
        file_path + ".tmp", format="mp3", loglevel="quiet"
    ).run()
    copy_input_file = file_path + ".tmp"

    # cut the audio file
    ffmpeg.input(copy_input_file, format="mp3").filter(
        "atrim", duration=duration
    ).output(file_path, loglevel="quiet").run(overwrite_output=True)

    # remove the copied file
    os.remove(copy_input_file)
