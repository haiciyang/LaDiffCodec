import torch
import torch.nn as nn

import typing as tp

from audiocraft import quantization as qt
from audiocraft.models import CompressionModel

class AE(CompressionModel):
    """Autoencoder model having encodec architecture operating on the raw waveform.

    Args:
        encoder (nn.Module): Encoder network.
        decoder (nn.Module): Decoder network.
        frame_rate (int): Frame rate for the latent representation.
        sample_rate (int): Audio sample rate.
        channels (int): Number of audio channels.
        causal (bool): Whether to use a causal version of the model.
        renormalize (bool): Whether to renormalize the audio before running the model.
    """
    # we need assignment to override the property in the abstract class,
    # I couldn't find a better way...
    frame_rate: float = 0
    sample_rate: int = 0
    channels: int = 0

    def __init__(self,
                 encoder: nn.Module,
                 decoder: nn.Module,
                #  quantizer: None,
                 frame_rate: int,
                 sample_rate: int,
                 channels: int,
                 causal: bool = False,
                 renormalize: bool = False):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.frame_rate = frame_rate
        self.sample_rate = sample_rate
        self.channels = channels
        self.renormalize = renormalize
        self.causal = causal
        if self.causal:
            # we force disabling here to avoid handling linear overlap of segments
            # as supported in original EnCodec codebase.
            assert not self.renormalize, 'Causal model does not support renormalize'

    @property
    def total_codebooks(self):
        """Total number of quantizer codebooks available."""
        return 0

    @property
    def num_codebooks(self):
        """Active number of codebooks used by the quantizer."""
        return 0

    def set_num_codebooks(self, n: int):
        """Set the active number of codebooks used by the quantizer."""
        pass

    @property
    def cardinality(self):
        """Cardinality of each codebook."""
        return 0
    
    def preprocess(self, x: torch.Tensor) -> tp.Tuple[torch.Tensor, tp.Optional[torch.Tensor]]:
        scale: tp.Optional[torch.Tensor]
        if self.renormalize:
            mono = x.mean(dim=1, keepdim=True)
            volume = mono.pow(2).mean(dim=2, keepdim=True).sqrt()
            scale = 1e-8 + volume
            x = x / scale
            scale = scale.view(-1, 1)
        else:
            scale = None
        return x, scale

    def postprocess(self,
                    x: torch.Tensor,
                    scale: tp.Optional[torch.Tensor] = None) -> torch.Tensor:
        if scale is not None:
            assert self.renormalize
            x = x * scale.view(-1, 1, 1)
        return x

    def forward(self, x: torch.Tensor) -> qt.QuantizedResult:
        assert x.dim() == 3
        length = x.shape[-1]
        x, scale = self.preprocess(x)

        emb = self.encoder(x)
        out = self.decoder(emb)

        # remove extra padding added by the encoder and decoder
        assert out.shape[-1] >= length, (out.shape[-1], length)
        out = out[..., :length]

        out = self.postprocess(out, scale)

        return qt.QuantizedResult(out, None, torch.tensor([0.]), penalty=None)

    def encode(self, x: torch.Tensor) -> tp.Tuple[torch.Tensor, tp.Optional[torch.Tensor]]:
        """Encode the given input tensor to quantized representation along with scale parameter.

        Args:
            x (torch.Tensor): Float tensor of shape [B, C, T]

        Returns:
            codes, scale (tuple of torch.Tensor, torch.Tensor): Tuple composed of:
                codes: a float tensor of shape [B, K, T] with K the number of codebooks used and T the timestep.
                scale: a float tensor containing the scale for audio renormalization.
        """
        assert x.dim() == 3
        x, scale = self.preprocess(x)
        emb = self.encoder(x)
        # print(codecs.shape, type(codes), codecs[0, :, 2])
        # fake()
        return emb, scale

    def decode(self, emb: torch.Tensor, scale: tp.Optional[torch.Tensor] = None):
        """Decode the given codes to a reconstructed representation, using the scale to perform
        audio denormalization if needed.

        Args:
            codes (torch.Tensor): Int tensor of shape [B, K, T]
            scale (torch.Tensor, optional): Float tensor containing the scale value.

        Returns:
            out (torch.Tensor): Float tensor of shape [B, C, T], the reconstructed audio.
        """
        out = self.decoder(emb)
        out = self.postprocess(out, scale)
        # out contains extra padding added by the encoder and decoder
        return out

    def decode_latent(self, emb: torch.Tensor):
        """Decode from the discrete codes to continuous latent space."""
        return emb

    
