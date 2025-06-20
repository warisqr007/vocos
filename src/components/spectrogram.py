import typing as tp
from dataclasses import dataclass


import torch
import math
import numpy as np
import torch.nn as nn
from torch import Tensor
import torchaudio.functional as F
from torchaudio.transforms import MelScale

from .streaming import StreamingModule
from .conv import get_extra_padding_for_conv1d, pad1d


class LinearSpectrogram(nn.Module):
    def __init__(self, n_fft=1024, win_length=1024, hop_length=320, mode="pow2_sqrt"):
        """
        Initializes the streaming spectrogram module.
        
        Parameters:
            n_fft (int): Number of FFT points.
            win_length (int): Window length (in samples).
            hop_length (int): Hop length (in samples).
            mode (str): Calculation mode. "pow2_sqrt" computes magnitude as sqrt(sum(squared)).
        """
        super().__init__()
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.mode = mode
        
        # Register the Hann window as a buffer.
        self.register_buffer("window", torch.hann_window(win_length), persistent=False)
        

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        """
        Computes the spectrogram from an input waveform chunk.
        
        Parameters:
            y (Tensor): Input waveform of shape (batch, time).
                                      
        Returns:
            spec (Tensor): Spectrogram tensor of shape (batch, n_frames, n_fft//2+1).
        """
        batch_size, n_samples = y.shape
        device = y.device

        
        total_length = y.size(1)
        # Compute the number of full frames available.
        n_frames = (total_length - self.win_length) // self.hop_length + 1

        if n_frames <= 0:
            # Not enough samples to form one full frame: update state and return an empty spectrogram.
            return torch.empty(batch_size, self.n_fft // 2 + 1, 0, device=device)

        
        # Determine the number of samples used in forming complete frames.
        used_length = (n_frames - 1) * self.hop_length + self.win_length

        # if used_length < total_length:
        #     warnings.warn(f"Extra {total_length - used_length} samples will be discarded to form complete frames.")
        
        y = y[:, :used_length]
        
        # Extract overlapping frames using unfolding.
        # The resulting shape is (batch, n_frames, win_length)
        frames = y.unfold(dimension=1, size=self.win_length, step=self.hop_length)
        
        # Apply the window to each frame.
        frames = frames * self.window
        
        # Compute the real FFT on each frame.
        spec = torch.fft.rfft(frames, n=self.n_fft)
        
        # If using "pow2_sqrt" mode, compute the magnitude spectrogram.
        if self.mode == "pow2_sqrt":
            # Compute sqrt(real^2 + imag^2) with epsilon for numerical stability.
            spec = torch.sqrt(spec.real.pow(2) + spec.imag.pow(2) + 1e-6)
        
        spec = spec.transpose(1, 2)  # (batch, n_fft//2+1, n_frames)
        return spec



class LogMelSpectrogram(nn.Module):
    def __init__(
        self,
        sample_rate=16000,
        n_fft=1024,
        win_length=1024,
        hop_length=320,
        n_mels=128,
        f_min=0.0,
        f_max=None,
    ):
        super().__init__()

        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.f_min = f_min
        self.f_max = f_max or float(sample_rate // 2)

        self.spectrogram = LinearSpectrogram(n_fft, win_length, hop_length)

        fb = F.melscale_fbanks(
            n_freqs=self.n_fft // 2 + 1,
            f_min=self.f_min,
            f_max=self.f_max,
            n_mels=self.n_mels,
            sample_rate=self.sample_rate,
            norm="slaney",
            mel_scale="slaney",
        )
        self.register_buffer(
            "fb",
            fb,
            persistent=False,
        )

    def compress(self, x: Tensor) -> Tensor:
        return torch.log(torch.clamp(x, min=1e-5))

    def decompress(self, x: Tensor) -> Tensor:
        return torch.exp(x)

    def apply_mel_scale(self, x: Tensor) -> Tensor:
        return torch.matmul(x.transpose(-1, -2), self.fb).transpose(-1, -2)

    def forward(
        self, x: Tensor, return_linear: bool = False, sample_rate: int = None
    ) -> Tensor:
        x = x.squeeze(1)
        if sample_rate is not None and sample_rate != self.sample_rate:
            x = F.resample(x, orig_freq=sample_rate, new_freq=self.sample_rate)

        linear = self.spectrogram(x)

        if linear.shape[-1] != 0:
            mel = self.apply_mel_scale(linear)
            mel = self.compress(mel)
        else:
            # Not enough samples to form one full frame: return an empty spectrogram.
            mel = torch.empty(
                linear.shape[0], self.n_mels, 0, device=linear.device, dtype=linear.dtype
            )

        compressed_linear = self.compress(linear) if linear.shape[-1] != 0 else linear
        if return_linear:
            return mel, compressed_linear

        return mel


@dataclass
class _StreamingSpecState:
    previous: torch.Tensor | None = None

    def reset(self):
        self.previous = None


class RawStreamingLogMelSpectrogram(LogMelSpectrogram, StreamingModule[_StreamingSpecState]):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        assert (
            self.hop_length <= self.win_length
        ), "stride must be less than kernel_size."

    def _init_streaming_state(self, batch_size: int) -> _StreamingSpecState:
        return _StreamingSpecState()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        stride = self.hop_length
        kernel = self.win_length
        if self._streaming_state is None:
            return super().forward(input)
        else:
            # Due to the potential overlap, we might have some cache of the previous time steps.
            previous = self._streaming_state.previous
            if previous is not None:
                input = torch.cat([previous, input], dim=-1)
            B, C, T = input.shape
            # We now compute the number of full convolution frames, i.e. the frames
            # that are ready to be computed.
            num_frames = max(0, int(math.floor((T - kernel) / stride) + 1))
            offset = num_frames * stride
            # We will compute `num_frames` outputs, and we are advancing by `stride`
            # for each of the frame, so we know the data before `stride * num_frames`
            # will never be used again.
            self._streaming_state.previous = input[..., offset:]
            if num_frames > 0:
                input_length = (num_frames - 1) * stride + kernel
                out = super().forward(input[..., :input_length])
            else:
                # Not enough data as this point to output some new frames.
                out = torch.empty(
                    B, self.n_mels, 0, device=input.device, dtype=input.dtype
                )
            return out



@dataclass
class _StreamingLogMelSpecState:
    padding_to_add: int
    original_padding_to_add: int

    def reset(self):
        self.padding_to_add = self.original_padding_to_add


class StreamingLogMelSpectrogram(StreamingModule[_StreamingLogMelSpecState]):
    """LogMelSpectrogram with some builtin handling of asymmetric or causal padding
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        n_fft: int = 1024,
        win_length: int = 1024,
        hop_length: int = 320,
        n_mels: int = 128,
        f_min: float = 0.0,
        f_max: float = None,
        causal: bool = False,
        pad_mode: str = "reflect",
    ):
        super().__init__()

        self.conv = RawStreamingLogMelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            n_mels=n_mels,
            f_min=f_min,
            f_max=f_max,
        )
        self.causal = causal
        self.pad_mode = pad_mode

    @property
    def _stride(self) -> int:
        return self.conv.hop_length

    @property
    def _kernel_size(self) -> int:
        return self.conv.win_length

    @property
    def _effective_kernel_size(self) -> int:
        return self._kernel_size

    @property
    def _padding_total(self) -> int:
        return self._effective_kernel_size - self._stride

    def _init_streaming_state(self, batch_size: int) -> _StreamingLogMelSpecState:
        assert self.causal, "streaming is only supported for causal convs"
        return _StreamingLogMelSpecState(self._padding_total, self._padding_total)

    def forward(self, x):
        B, C, T = x.shape
        padding_total = self._padding_total
        extra_padding = get_extra_padding_for_conv1d(
            x, self._effective_kernel_size, self._stride, padding_total
        )
        state = self._streaming_state
        if state is None:
            if self.causal:
                # Left padding for causal
                x = pad1d(x, (padding_total, extra_padding), mode=self.pad_mode)
            else:
                # Asymmetric padding required for odd strides
                padding_right = padding_total // 2
                padding_left = padding_total - padding_right
                x = pad1d(
                    x, (padding_left, padding_right + extra_padding), mode=self.pad_mode
                )
        else:
            if state.padding_to_add > 0 and x.shape[-1] > 0:
                x = pad1d(x, (state.padding_to_add, 0), mode=self.pad_mode)
                state.padding_to_add = 0
        return self.conv(x)


class OverlapAdd1d(nn.Module):
    """
    Fixed ConvTranspose1d that performs overlap-add:
      in_channels = win_length, out_channels = 1, kernel=win_length, stride=hop.
    """
    def __init__(self, win_length: int, hop: int):
        super().__init__()
        self.deconv = nn.ConvTranspose1d(
            in_channels=win_length,
            out_channels=1,
            kernel_size=win_length,
            stride=hop,
            bias=False,
        )
        # build identity‑impulse weights
        w = torch.zeros(win_length, 1, win_length)
        for c in range(win_length):
            w[c, 0, c] = 1.0
        self.deconv.weight.data.copy_(w)
        self.deconv.weight.requires_grad_(False)

    def forward(self, x: Tensor) -> Tensor:
        # x: (B, win_length, F) → y: (B, 1, (F-1)*hop + win_length)
        y = self.deconv(x)
        return y.squeeze(1)  # → (B, T_out)


@dataclass
class _StreamingISTFTState:
    prev_buffer: Tensor

    def reset(self):
        self.prev_buffer.zero_()


class StreamingISTFT(StreamingModule[_StreamingISTFTState]):
    """
    Streaming ISTFT via overlap-add:
      - inverse FFT per frame
      - window multiplication
      - overlap-add using a fixed ConvTranspose1d
      - carry tail for next chunk
    """
    def __init__(
        self,
        n_fft: int,
        hop_length: int,
        win_length: int | None = None,
    ):
        super().__init__()
        self.n_fft = n_fft
        self.hop = hop_length
        self.win_length = win_length or n_fft

        # hann window
        win = torch.hann_window(self.win_length)
        self.register_buffer("window", win, persistent=False)

        # overlap‑add helper
        self.overlap_add = OverlapAdd1d(self.win_length, self.hop)

        # how many samples to carry
        self.tail = self.win_length - self.hop
        if self.tail < 0:
            raise ValueError("hop_length must be <= win_length")

    def _init_streaming_state(self, batch_size: int) -> _StreamingISTFTState:
        buf = torch.zeros(batch_size, self.tail, device=self.window.device)
        return _StreamingISTFTState(prev_buffer=buf)

    def forward(self, S: Tensor) -> Tensor:
        """
        Args:
          S: complex STFT chunk, shape (B, n_fft//2+1, F_frames)
        Returns:
          time-domain chunk, shape (B, F_frames * hop_length)
        """
        B, n_freq, F = S.shape

        if F == 0:
            # Not enough samples to form one full frame: return an empty tensor.
            return torch.empty(B, 0, device=S.device, dtype=S.dtype)

        state = self._streaming_state
        # if state is None:
        #     # no streaming state, just do a regular ISTFT
        #     return torch.istft(
        #         S,
        #         n_fft=self.n_fft,
        #         hop_length=self.hop,
        #         win_length=self.win_length,
        #         window=self.window,
        #         center=False,
        #     )

        # (B, F, n_fft//2+1) -> real frames (B, F, n_fft)
        spec = S.transpose(1, 2)
        frames = torch.fft.irfft(spec, n=self.n_fft)
        # window + truncate
        frames = frames[..., : self.win_length] * self.window

        # (B, F, win) -> (B, win, F)
        x = frames.transpose(1, 2)

        # overlap-add in one CUDA kernel
        out = self.overlap_add(x)  # (B, F*hop + tail)

        if state is not None:
            # add carried‑over overlap
            out[:, : self.tail] += self._streaming_state.prev_buffer

            # slice off ready frames and save new tail
            ready = out[:, : F * self.hop]
            tail = out[:, F * self.hop :]
            self._streaming_state.prev_buffer = tail
        else:
            ready = out[:, : F * self.hop]
        return ready