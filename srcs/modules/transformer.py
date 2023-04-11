import math
import torch
from torch import nn

class ConvLayerNorm(nn.LayerNorm):
    """
    Convolution-friendly LayerNorm that moves channels to last dimensions
    before running the normalization and moves them back to original position right after.
    """
    def __init__(self, normalized_shape, **kwargs):
        super().__init__(normalized_shape, **kwargs)

    def forward(self, x):
        x = einops.rearrange(x, 'b ... t -> b t ...')
        x = super().forward(x)
        x = einops.rearrange(x, 'b t ... -> b ... t')
        return

class ConvLinear(nn.Linear):
    """
    Convolution-friendly LayerNorm that moves channels to last dimensions
    before running the normalization and moves them back to original position right after.
    """
    def __init__(self, in_features, out_features, **kwargs):
        super().__init__(in_features, out_features, **kwargs)

    def forward(self, x):
        x = einops.rearrange(x, 'b ... t -> b t ...')
        x = super().forward(x)
        x = einops.rearrange(x, 'b t ... -> b ... t')
        return


class NoiseEncoding(nn.Module):
  """Sinusoidal noise encoding block."""

  def __init__(self, channels):
    super().__init__()
    self.channels = channels

  def forward(self, noise):
    # noise.shape = (batch_size, 1)
    # channels.shape = ()
    noise = noise.squeeze(-1)
    assert len(noise.shape) == 1
    half_dim = self.channels // 2
    emb = math.log(10000) / float(half_dim - 1)
    emb = torch.exp(torch.arange(half_dim) * -emb).to(noise.device)
    emb = 5000 * noise[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if self.channels % 2 == 1:
      emb = torch.pad(emb, [[0, 0], [0, 1]])
    assert emb.shape == (noise.shape[0], self.channels)
    return emb


class TransformerPositionalEncoding(nn.Module):
  """Transformer positional encoding block."""

  def __init__(self, channels):
    super().__init__()
    self.channels = channels

  def forward(self, timesteps):
    # timesteps.shape = (seq_len,)
    # channels.shape = ()

    assert len(timesteps.shape) == 1
    half_dim = self.channels // 2

    emb = math.log(10000) / float(half_dim - 1)
    emb = torch.exp(torch.arange(half_dim) * -emb)

    emb = timesteps[:, None] * emb[None, :] #  ( L, C//2,)
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1) 

    if self.channels % 2 == 1:
      emb = torch.pad(emb, [[0, 0], [0, 1]])
    assert emb.shape == (timesteps.shape[0], self.channels )

    return emb


class DenseFiLM(nn.Module):
  """Feature-wise linear modulation (FiLM) generator."""

  def __init__(self, channels, out_channels, sequence=False, cond=False):
   
    super().__init__()

    self.sequence = sequence

    self.net = nn.Sequential(
      NoiseEncoding(channels),
      nn.Linear(channels, channels*4),
      nn.SiLU(),
      nn.Linear(channels*4, channels*4),
    )

    self.condNet = nn.Sequential(
      nn.Linear(channels, channels*4),
      nn.SiLU(),
      nn.Linear(channels*4, channels*4),
    )

    self.output_scale = nn.Linear(channels*4, out_channels)
    self.output_shift = nn.Linear(channels*4, out_channels)

  def forward(self, pos, cond=None):

    # pos.shape - (batch_size, 1)

    pos_encoding = self.net(pos)

    if self.sequence:
      pos_encoding = pos_encoding[:, None, :]
    if cond is not None:
      pos_encoding = pos_encoding[:, None, :]
      assert pos_encoding.shape[-1] == cond.shape[-1]
      pos_encoding = pos_encoding + self.condNet(cond)

    scale = self.output_scale(pos_encoding)
    shift = self.output_shift(pos_encoding)

    return scale, shift

class DenseResBlock(nn.Module):
  """Fully-connected residual block."""

  def __init__(self, inp_dimension, out_dimension):
    super().__init__()

    if inp_dimension != out_dimension:
      self.cmp_layer = nn.Linear(inp_dimension, out_dimension)
    
    self.layernorm = nn.LayerNorm(inp_dimension)
    self.fc1 = nn.Sequential(
      nn.SiLU(),
      nn.Linear(inp_dimension, out_dimension),
    )
    self.fc2 = nn.Sequential(
      nn.SiLU(),
      nn.Linear(inp_dimension, out_dimension),
    )

  def featurewiseAffine(self, x, scale=1., shift=0.):
    
    if not len(scale.shape) == 3:
      scale = scale.unsqueeze(1)
      shift = shift.unsqueeze(1)

    return scale * x + shift

  def forward(self, x, scale, shift):

    # x_out = self.net(x)
    x_out = self.layernorm(x)
    x_out = self.featurewiseAffine(x_out, scale, shift)

    
    x_out = self.fc1(x_out)

    x_out = self.layernorm(x_out)
    x_out = self.featurewiseAffine(x_out, scale, shift)
    x_out = self.fc2(x_out)

    if x.shape[-1] == x_out.shape[-1]:
      shortcut = x 
    else:
      shortcut = self.cmp_layer(x)

    return x_out + shortcut


class SelfMultiHeadAttention(nn.MultiheadAttention):
  def __init__(self, emb_dims=128, num_heads=8):

    super().__init__(emb_dims, num_heads)
  
  def forward(self, x):
    x, _ = super().forward(x, x, x)
    return x


class TransformerEncoderBlock(nn.Module):
  def __init__(self, emb_dims=128, mlp_dims=2048, num_heads=8, ):

    super().__init__()

    self.attentionBlock = nn.Sequential(
      nn.LayerNorm(emb_dims),
      SelfMultiHeadAttention(emb_dims, num_heads)
    )

    self.linearBlock = nn.Sequential(
      nn.LayerNorm(emb_dims),
      nn.Linear(emb_dims, mlp_dims),
      nn.GELU(),
      nn.Linear(mlp_dims, emb_dims)
    )
  
  def forward(self, x):

    x = x + self.attentionBlock(x)
    x = x + self.linearBlock(x)

    return x

class NoiseCondResidual(nn.Module):
  def __init__(self, channels=128, out_channels=2048, sequence=False):
    super().__init__()

    self.denseFiLM = DenseFiLM(channels, out_channels, sequence)
    self.denseBlock = DenseResBlock(channels, out_channels)


  def forward(self, x, t):
    # t - (Bt,)
    bt = t.shape[0]
    t = t.reshape(bt, 1)
    scale, shift = self.denseFiLM(t)
    x = self.denseBlock(x, scale, shift)

    return x


class TransformerDDPM(nn.Module):
  """Transformer-based diffusion model."""

  def __init__(self,
            rep_dims = 128,
            emb_dims = 128, 
            mlp_dims= 2048,
            num_layers= 6,
            num_heads= 8,
            num_mlp_layers=2,
            self_condition=False
            ):

    super().__init__()

    self.channels = rep_dims
    self.self_condition = self_condition
    self.rep_dims = rep_dims

    self.pos_encoding = TransformerPositionalEncoding(emb_dims)
    self.first_layer = nn.Linear(rep_dims, emb_dims)
  
    model = []
    for _ in range(num_layers):
      model += [TransformerEncoderBlock(emb_dims, mlp_dims, num_heads)]
    
    model += [
      nn.LayerNorm(emb_dims),
      nn.Linear(emb_dims, mlp_dims)
    ]

    self.encoder = nn.Sequential(*model)

    self.condBlocks = nn.ModuleList([])
    # 2 noise conditioned residual blocks
    for _ in range(num_mlp_layers):
      self.condBlocks.append(NoiseCondResidual(mlp_dims, mlp_dims))

    models = []
    models+= [
      nn.LayerNorm(mlp_dims),
      nn.Linear(mlp_dims, rep_dims)
    ]
    self.output_layers = nn.Sequential(*models)

  def forward(self, x, t, *args):

    x = x.transpose(1, 2)

    batch_size, seq_len, data_channels = x.shape
    assert data_channels == self.rep_dims

    # Positinoal encoding
    temb = self.pos_encoding(torch.arange(seq_len)).unsqueeze(0).to(x.device)
    x = self.first_layer(x)
    x = x + temb

    # Transformer encoder
    x = self.encoder(x)

    # Add noise condition
    for net in self.condBlocks:
      x = net(x, t)
    
    # Generate reverse process output
    output = self.output_layers(x)
    output = output.transpose(1, 2)

    return output


if __name__ == '__main__':

  transformer = TransformerDDPM()
  x = torch.rand(5, 100, 128)
  t = torch.rand(5, 1)

  output = transformer(x, t)

  print(output.shape
  )








  # def apply(self,
  #           inputs,
  #           t,
  #           num_layers=6,
  #           num_heads=8,
  #           num_mlp_layers=2,
  #           mlp_dims=2048):

  #   batch_size, seq_len, data_channels = inputs.shape

  #   x = inputs
  #   embed_channels = 128
  #   temb = TransformerPositionalEncoding(torch.arange(seq_len), embed_channels)
  #   temb = temb[None, :, :]
  #   assert temb.shape[1:] == (seq_len, embed_channels), temb.shape

  #   x = nn.Dense(x, embed_channels)

  #   x = x + temb
  #   for _ in range(num_layers):
  #     shortcut = x
  #     x = nn.LayerNorm(x)
  #     x = nn.SelfAttention(x, num_heads=num_heads)
  #     x = x + shortcut
  #     shortcut2 = x
  #     x = nn.LayerNorm(x)
  #     x = nn.Dense(x, mlp_dims)
  #     x = nn.gelu(x)
  #     x = nn.Dense(x, embed_channels)
  #     x = x + shortcut2

  #   x = nn.LayerNorm(x)
  #   x = nn.Dense(x, mlp_dims)

  #   for _ in range(num_mlp_layers):
  #     scale, shift = DenseFiLM(t.squeeze(-1), 128, mlp_dims, sequence=True)
  #     x = DenseResBlock(x, mlp_dims, scale=scale, shift=shift)

  #   x = nn.LayerNorm(x)
  #   x = nn.Dense(x, data_channels)
  #   return 