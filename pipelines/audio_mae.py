import math
from typing import Tuple

import timm
import torch
import torchaudio
import torchaudio.transforms as transforms
from einops import rearrange
from timm.models.vision_transformer import VisionTransformer
from torchaudio.compliance import kaldi
from transformers import PretrainedConfig, PreTrainedModel


class AudioMAEConfig(PretrainedConfig):
    model_type = "audiomae"

    def __init__(
        self,
        img_size: Tuple[int, int] = (1024, 128),
        in_chans: int = 1,
        num_classes: int = 0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.img_size = img_size
        self.in_chans = in_chans
        self.num_classes = num_classes


class AudioMAEEncoder(VisionTransformer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.MEAN = -4.2677393
        self.STD = 4.5689974
        self.WINDOW_SIZE = 10.0  # 10 seconds window
        self.STRIDE = 10.0  # 10 second stride

    def load_wav_file(self, file_path: str) -> torch.FloatTensor:
        """Load and prepare audio file."""
        audio, sample_rate = torchaudio.load(file_path)

        # Convert stereo to mono if necessary
        if audio.shape[0] > 1:
            audio = torch.mean(audio, dim=0, keepdim=True)

        # Resample to 16kHz if needed
        if sample_rate != 16000:
            converter = transforms.Resample(orig_freq=sample_rate, new_freq=16000)
            audio = converter(audio)

        return audio

    def load_wav_array(
        self, audio: torch.FloatTensor, sample_rate: int
    ) -> torch.FloatTensor:
        """Prepare audio array."""
        # Convert stereo to mono if necessary
        if audio.shape[0] > 1:
            audio = torch.mean(audio, dim=0, keepdim=True)

        # Resample to 16kHz if needed
        if sample_rate != 16000:
            converter = transforms.Resample(orig_freq=sample_rate, new_freq=16000)
            audio = converter(audio)

        return audio

    def extract_windows(self, waveform: torch.FloatTensor) -> torch.FloatTensor:
        """Extract overlapping windows from the waveform.

        Args:
            waveform: Input audio tensor of shape (1, samples)

        Returns:
            Tensor of windows of shape (num_windows, 1, window_samples)
        """
        sample_rate = 16000
        window_samples = int(self.WINDOW_SIZE * sample_rate)  # 160000 samples for 10s
        stride_samples = int(self.STRIDE * sample_rate)  # 16000 samples for 1s

        # Get total number of samples
        total_samples = waveform.shape[1]

        # Calculate number of complete windows
        num_windows = max(
            1, math.floor((total_samples - window_samples) / stride_samples)
        )

        # Initialize tensor to store windows
        windows = []

        # Extract windows
        for i in range(num_windows):
            start_idx = i * stride_samples
            end_idx = start_idx + window_samples

            # If we would exceed the audio length, break
            if end_idx > total_samples:
                break

            window = waveform[:, start_idx:end_idx]
            windows.append(window)

        # Stack all windows
        if windows:
            return torch.stack(windows)  # (num_windows, 1, window_samples)
        else:
            # If the audio is shorter than the window size, pad it
            padded = torch.nn.functional.pad(
                waveform, (0, window_samples - total_samples), mode="constant"
            )
            return padded.unsqueeze(0)  # Add window dimension

    def waveform_to_melspec(self, waveform: torch.FloatTensor) -> torch.FloatTensor:
        """Convert waveform windows to mel spectrograms."""
        mel_spectrogram = kaldi.fbank(
            waveform,
            num_mel_bins=128,
            frame_length=25.0,
            frame_shift=10.0,
            htk_compat=True,
            use_energy=False,
            sample_frequency=16000,
            window_type="hanning",
            dither=0.0,
        )

        # Ensure the output shape matches 1024x128
        expected_frames = 1024
        current_frames = mel_spectrogram.shape[0]

        if current_frames > expected_frames:
            mel_spectrogram = mel_spectrogram[:expected_frames, :]
        elif current_frames < expected_frames:
            padding = expected_frames - current_frames
            mel_spectrogram = torch.nn.functional.pad(
                mel_spectrogram, (0, 0, 0, padding)
            )

        # Scale
        mel_spectrogram = (mel_spectrogram - self.MEAN) / (self.STD * 2)
        return mel_spectrogram

    @torch.no_grad()
    def encode(
        self, file: str | torch.Tensor, device, sample_rate: int = None
    ) -> torch.FloatTensor:
        """Encode audio file using sliding windows approach."""
        self.eval()

        if isinstance(file, str):
            # Load the audio file
            waveform = self.load_wav_file(file)
        else:
            if sample_rate is None:
                raise ValueError("Sample rate must be provided when passing a tensor")
            waveform = self.load_wav_array(file, sample_rate)

        # Extract windows
        windows = self.extract_windows(waveform)

        # Process each window
        window_embeddings = []
        for window in windows:
            # Convert to mel spectrogram
            melspec = self.waveform_to_melspec(window)  # (1024, 128)
            melspec = melspec[None, None, :, :]  # (1, 1, 1024, 128)

            # Get embeddings
            z = self.forward_features(melspec.to(device)).cpu()  # (1, 1+n, d)
            z = z[:, 1:, :]  # Remove [CLS] token

            # Reshape embeddings
            b, c, w, h = melspec.shape
            wprime = round(w / self.patch_embed.patch_size[0])
            hprime = round(h / self.patch_embed.patch_size[1])
            z = rearrange(z, "b (w h) d -> b d h w", h=hprime)  # (1, d, h', w')

            window_embeddings.append(z[0])  # Remove batch dimension

        # Average all window embeddings
        if len(window_embeddings) > 0:
            final_embedding = torch.stack(window_embeddings).mean(dim=0)  # (d, h', w')
        else:
            raise ValueError("No valid windows were extracted from the audio file")

        return final_embedding


class PretrainedAudioMAEEncoder(PreTrainedModel):
    config_class = AudioMAEConfig

    def __init__(self, config):
        super().__init__(config)
        self.encoder = AudioMAEEncoder(
            img_size=config.img_size,
            in_chans=config.in_chans,
            num_classes=config.num_classes,
        )

    def forward(self, file_path: str, sample_rate: int = None):
        device = self.device
        return self.encoder.encode(file_path, device, sample_rate)

    def load_model(self):
        pretrained_model = timm.create_model(
            "hf_hub:gaunernst/vit_base_patch16_1024_128.audiomae_as2m", pretrained=True
        )

        pretrained_model = pretrained_model.eval()
        pretrained_state_dict = pretrained_model.state_dict()
        new_keys = []

        for k in pretrained_state_dict.keys():
            new_keys.append("encoder." + k)
        pretrained_state_dict = {
            new_keys[i]: v for i, (k, v) in enumerate(pretrained_state_dict.items())
        }

        self.load_state_dict(pretrained_state_dict)

        return self
