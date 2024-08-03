"""
Module to extract features from the audio data.
"""

import essentia
import essentia.standard as es
import essentia.streaming as ess
import librosa
import numpy as np
from essentia.standard import ChordsDetection


class FeatureExtractor:
    """Feature extractor for the audio data."""

    def extract_tempo(self, audio_path: str) -> float:
        """Extract the tempo of the audio data.

        Args:
            audio_path (str): the path to the audio file.

        Returns:
            float: the tempo of the audio data.
        """
        y, sr = librosa.load(audio_path)
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        tempo = librosa.beat.tempo(onset_envelope=onset_env, sr=sr)
        return tempo[0]

    def extract_key(self, audio_path: str) -> str:
        """Extract the key of the audio data.

        Args:
            audio_path (str): the path to the audio file.

        Returns:
            str: the key of the audio data.
        """
        y, sr = librosa.load(audio_path)

        chromagram = librosa.feature.chroma_stft(y=y, sr=sr)
        mean_chroma = np.mean(chromagram, axis=1)

        chroma_to_key = [
            "C",
            "C#",
            "D",
            "D#",
            "E",
            "F",
            "F#",
            "G",
            "G#",
            "A",
            "A#",
            "B",
        ]

        estimated_key_index = np.argmax(mean_chroma)
        estimated_key = chroma_to_key[estimated_key_index]

        return estimated_key

    def extract_downbeats(self, audio_path: str) -> list:
        """Extract the downbeats of the audio data.

        Args:
            audio_path (str): the path to the audio file.

        Returns:
            list: the downbeats of the audio data.
        """
        audio = es.MonoLoader(filename=audio_path)()

        rhythm_extractor = es.RhythmExtractor2013(method="multifeature")

        _, beats, _, _, _ = rhythm_extractor(audio)
        return beats

    def extract_chords(self, audio_path: str) -> list:
        """Extract the chords of the audio data.

        Args:
            audio_path (str): the path to the audio file.

        Returns:
            list: the chords of the audio data.
        """

        loader = ess.MonoLoader(filename=audio_path)
        framecutter = ess.FrameCutter(
            frameSize=4096, hopSize=2048, silentFrames="noise"
        )
        windowing = ess.Windowing(type="blackmanharris62")
        spectrum = ess.Spectrum()
        spectralpeaks = ess.SpectralPeaks(
            orderBy="magnitude",
            magnitudeThreshold=0.00001,
            minFrequency=20,
            maxFrequency=3500,
            maxPeaks=60,
        )

        # Use default HPCP parameters for plots.
        # However we will need higher resolution and custom parameters for better Key estimation.
        hpcp = ess.HPCP()

        # Use pool to store data.
        pool = essentia.Pool()

        # Connect streaming algorithms.
        loader.audio >> framecutter.signal
        framecutter.frame >> windowing.frame >> spectrum.frame
        spectrum.spectrum >> spectralpeaks.spectrum
        spectralpeaks.magnitudes >> hpcp.magnitudes
        spectralpeaks.frequencies >> hpcp.frequencies
        hpcp.hpcp >> (pool, "tonal.hpcp")

        # Run streaming network.
        essentia.run(loader)

        # Using a 2 seconds window over HPCP matrix to estimate chords
        chords, _ = ChordsDetection(hopSize=2048, windowSize=2)(pool["tonal.hpcp"])
        return chords

    def print_features(self, audio_path: str) -> str:
        """Extracts the features of the audio data.

        Args:
            audio_path (str): the path to the audio file.

        Returns:
            str: a string that contain the tempo, downbeats, key, and chords of the audio data.
        """
        tempo = self.extract_tempo(audio_path)
        downbeats = self.extract_downbeats(audio_path)
        key = self.extract_key(audio_path)
        chords = self.extract_chords(audio_path)

        out_str = f" Tempo: {round(tempo,1)}BPM, \
                Key of the music: {key}, \
                20 firsts timestamp of Downbeats: {downbeats[:20]}, \
                20 firsts Chords (2 seconds windows): {chords[:20]}".strip()

        return out_str
