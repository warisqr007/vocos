from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class Lambdas:
    mel_loss: float = 15.0
    adv_feat_loss: float = 2.0
    adv_gen_loss: float = 1.0


@dataclass
class ISTFTHeadArgs:
    dim: int = 512
    n_fft: int = 1024
    hop_length: int = 320


@dataclass
class VocosVocoderArgs:
    input_channels: int = 80
    dim: int = 512
    mlp_ratio: float = 3.0
    kernel_size: int = 7
    dilation: int = 1
    norm: str = "weight_norm"
    causal: bool = True
    pad_mode: str = "constant"
    num_layers: int = 8
    layer_scale_init_value: float = 1e-6


@dataclass
class VocoderFeatureArgs:
    sample_rate: int = 16000
    n_fft: int = 1024
    win_length: int = 1024
    hop_length: int = 320
    n_mels: int = 80
    f_min: float = 0.0
    f_max: float = None
    causal: bool = True
    pad_mode: str = "constant"