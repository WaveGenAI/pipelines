""" 
Module for Descript Audio Codec (DAC) pipeline.
"""

import torch
from transformers import AutoModel


class DAC:
    """
    Descript Audio Codec  (DAC) pipeline.
    """

    def __init__(self):
        self.model = AutoModel.from_pretrained(
            "hance-ai/descript-audio-codec-44khz", trust_remote_code=True
        )
        device = "cpu" if not torch.cuda.is_available() else "cuda"
        self.model.to(device)

    def encode(self, audio: str) -> torch.Tensor:
        """Encode audio to codebooks index.

        Args:
            audio (str): Path to audio file.

        Returns:
            torch.Tensor: Codebooks index of shape (b, n_codebooks, n_frames).
        """
        _, s = self.model.encode(audio)
        return s

    def decode(self, s: torch.Tensor) -> torch.Tensor:
        """Decode codebooks index to audio.

        Args:
            s (torch.Tensor): Codebooks index of shape (b, n_codebooks, n_frames).

        Returns:
            torch.Tensor: Audio of shape (b, 1, n_samples).
        """
        return self.model.decode(s=s)

    def __getattr__(self, name):
        """Delegate attribute access to the underlying model."""
        return getattr(self.model, name)
