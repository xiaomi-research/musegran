"""Audio autoencoder with VAE bottleneck and chunked encode/decode support."""

import math
from typing import Any, Dict, Literal

import torch
from alias_free_torch import Activation1d
from dac.nn.layers import WNConv1d, WNConvTranspose1d
from torch import nn


# --- SnakeBeta activation ---

def _snake_beta(x, alpha, beta):
    return x + (1.0 / (beta + 1e-9)) * torch.sin(x * alpha).pow(2)


class SnakeBeta(nn.Module):
    def __init__(self, in_features, alpha=1.0, alpha_trainable=True, alpha_logscale=True):
        super().__init__()
        self.in_features = in_features
        self.alpha_logscale = alpha_logscale
        if self.alpha_logscale:
            self.alpha = nn.Parameter(torch.zeros(in_features) * alpha)
            self.beta = nn.Parameter(torch.zeros(in_features) * alpha)
        else:
            self.alpha = nn.Parameter(torch.ones(in_features) * alpha)
            self.beta = nn.Parameter(torch.ones(in_features) * alpha)
        self.alpha.requires_grad = alpha_trainable
        self.beta.requires_grad = alpha_trainable

    def forward(self, x):
        alpha = self.alpha.unsqueeze(0).unsqueeze(-1)
        beta = self.beta.unsqueeze(0).unsqueeze(-1)
        if self.alpha_logscale:
            alpha = torch.exp(alpha)
            beta = torch.exp(beta)
        return _snake_beta(x, alpha, beta)


# --- VAE Bottleneck ---

def vae_sample(mean, scale):
    stdev = nn.functional.softplus(scale) + 1e-4
    var = stdev * stdev
    logvar = torch.log(var)
    latents = torch.randn_like(mean) * stdev + mean
    kl = (mean * mean + var - logvar - 1).sum(1).mean()
    return latents, kl


class VAEBottleneck(nn.Module):
    """Gaussian VAE bottleneck with reparameterization trick."""

    def __init__(self):
        super().__init__()
        self.is_discrete = False

    def encode(self, x, return_info=False, **kwargs):
        mean, scale = x.chunk(2, dim=1)
        x, kl = vae_sample(mean, scale)
        if return_info:
            return x, {"kl": kl}
        return x

    def decode(self, x):
        return x


# --- Pretransform ---

class AutoencoderPretransform(nn.Module):
    """Frozen autoencoder used as a pretransform for latent diffusion."""

    def __init__(self, model, scale=1.0, model_half=False, iterate_batch=False):
        super().__init__()
        self.model = model
        self.model.requires_grad_(False).eval()
        self.scale = scale
        self.downsampling_ratio = model.downsampling_ratio
        self.io_channels = model.io_channels
        self.sample_rate = model.sample_rate
        self.model_half = model_half
        self.iterate_batch = iterate_batch
        self.encoded_channels = model.latent_dim
        self.enable_grad = False

        if self.model_half:
            self.model.half()

    def encode(self, x, **kwargs):
        if self.model_half:
            x = x.half()
            self.model.to(torch.float16)
        encoded = self.model.encode_audio(x, chunked=True, iterate_batch=self.iterate_batch, **kwargs)
        if self.model_half:
            encoded = encoded.float()
        return encoded / self.scale

    def decode(self, z, **kwargs):
        z = z * self.scale
        if self.model_half:
            z = z.half()
            self.model.to(torch.float16)
        decoded = self.model.decode_audio(z, chunked=True, iterate_batch=self.iterate_batch, **kwargs)
        if self.model_half:
            decoded = decoded.float()
        return decoded

    def load_state_dict(self, state_dict, strict=True):
        self.model.load_state_dict(state_dict, strict=strict)


# --- Building blocks ---

def get_activation(activation: Literal["elu", "snake", "none"], antialias=False, channels=None) -> nn.Module:
    if activation == "elu":
        act = nn.ELU()
    elif activation == "snake":
        act = SnakeBeta(channels)
    elif activation == "none":
        act = nn.Identity()
    else:
        raise ValueError(f"Unknown activation {activation}")

    if antialias:
        act = Activation1d(act)
    return act


class ResidualUnit(nn.Module):
    def __init__(self, in_channels, out_channels, dilation, use_snake=False, antialias_activation=False):
        super().__init__()
        self.dilation = dilation
        padding = (dilation * (7 - 1)) // 2

        self.layers = nn.Sequential(
            get_activation("snake" if use_snake else "elu", antialias=antialias_activation, channels=out_channels),
            WNConv1d(in_channels=in_channels, out_channels=out_channels,
                     kernel_size=7, dilation=dilation, padding=padding),
            get_activation("snake" if use_snake else "elu", antialias=antialias_activation, channels=out_channels),
            WNConv1d(in_channels=out_channels, out_channels=out_channels, kernel_size=1)
        )

    def forward(self, x):
        return x + self.layers(x)


class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, padding=None, use_snake=False, antialias_activation=False):
        super().__init__()
        self.layers = nn.Sequential(
            ResidualUnit(in_channels, in_channels, dilation=1, use_snake=use_snake),
            ResidualUnit(in_channels, in_channels, dilation=3, use_snake=use_snake),
            ResidualUnit(in_channels, in_channels, dilation=9, use_snake=use_snake),
            get_activation("snake" if use_snake else "elu", antialias=antialias_activation, channels=in_channels),
            WNConv1d(in_channels=in_channels, out_channels=out_channels,
                     kernel_size=2 * stride, stride=stride,
                     padding=math.ceil(stride / 2) if padding is None else padding),
        )

    def forward(self, x):
        return self.layers(x)


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, padding=None, output_padding=None,
                 use_snake=False, antialias_activation=False, use_nearest_upsample=False):
        super().__init__()

        if use_nearest_upsample:
            upsample_layer = nn.Sequential(
                nn.Upsample(scale_factor=stride, mode="nearest"),
                WNConv1d(in_channels=in_channels, out_channels=out_channels,
                         kernel_size=2 * stride, stride=1, bias=False, padding='same')
            )
        else:
            if output_padding is None:
                output_padding = 1 if stride % 2 == 1 else 0
            upsample_layer = WNConvTranspose1d(
                in_channels=in_channels, out_channels=out_channels,
                kernel_size=2 * stride, stride=stride,
                padding=math.ceil(stride / 2) if padding is None else padding,
                output_padding=output_padding)

        self.layers = nn.Sequential(
            get_activation("snake" if use_snake else "elu", antialias=antialias_activation, channels=in_channels),
            upsample_layer,
            ResidualUnit(out_channels, out_channels, dilation=1, use_snake=use_snake),
            ResidualUnit(out_channels, out_channels, dilation=3, use_snake=use_snake),
            ResidualUnit(out_channels, out_channels, dilation=9, use_snake=use_snake),
        )

    def forward(self, x):
        return self.layers(x)


# --- Encoder / Decoder ---

class OobleckEncoder(nn.Module):
    """Multi-scale strided convolutional encoder with optional SnakeBeta activation."""

    def __init__(self, in_channels=2, channels=128, latent_dim=32,
                 c_mults=[1, 2, 4, 8], strides=[2, 4, 8, 8],
                 padding=None, use_snake=False, antialias_activation=False):
        super().__init__()
        c_mults = [1] + c_mults
        self.depth = len(c_mults)

        layers = [WNConv1d(in_channels=in_channels, out_channels=c_mults[0] * channels, kernel_size=7, padding=3)]
        for i in range(self.depth - 1):
            pad = padding[i] if padding is not None else None
            layers.append(EncoderBlock(
                in_channels=c_mults[i] * channels, out_channels=c_mults[i + 1] * channels,
                stride=strides[i], padding=pad, use_snake=use_snake))

        layers += [
            get_activation("snake" if use_snake else "elu", antialias=antialias_activation, channels=c_mults[-1] * channels),
            WNConv1d(in_channels=c_mults[-1] * channels, out_channels=latent_dim, kernel_size=3, padding=1)
        ]
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class OobleckDecoder(nn.Module):
    """Multi-scale transposed convolutional decoder mirroring OobleckEncoder."""

    def __init__(self, out_channels=2, channels=128, latent_dim=32,
                 c_mults=[1, 2, 4, 8], strides=[2, 4, 8, 8],
                 padding=None, output_padding=None, use_snake=False,
                 antialias_activation=False, use_nearest_upsample=False, final_tanh=True):
        super().__init__()
        c_mults = [1] + c_mults
        self.depth = len(c_mults)

        layers = [WNConv1d(in_channels=latent_dim, out_channels=c_mults[-1] * channels, kernel_size=7, padding=3)]
        for i in range(self.depth - 1, 0, -1):
            layers.append(DecoderBlock(
                in_channels=c_mults[i] * channels, out_channels=c_mults[i - 1] * channels,
                stride=strides[i - 1], use_snake=use_snake,
                padding=padding[i - 1] if padding is not None else None,
                output_padding=output_padding[i - 1] if output_padding is not None else None,
                antialias_activation=antialias_activation,
                use_nearest_upsample=use_nearest_upsample))

        layers += [
            get_activation("snake" if use_snake else "elu", antialias=antialias_activation, channels=c_mults[0] * channels),
            WNConv1d(in_channels=c_mults[0] * channels, out_channels=out_channels, kernel_size=7, padding=3, bias=False),
            nn.Tanh() if final_tanh else nn.Identity()
        ]
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


# --- Audio Autoencoder ---

def _apply_fn(fn, x, iterate_batch):
    """Apply fn to x, optionally iterating over batch dimension."""
    if iterate_batch:
        return torch.cat([fn(x[i:i+1]) for i in range(x.shape[0])], dim=0)
    return fn(x)


class AudioAutoencoder(nn.Module):
    """Audio autoencoder supporting chunked encode/decode and optional VAE bottleneck."""

    def __init__(self, encoder, decoder, latent_dim, downsampling_ratio, sample_rate,
                 io_channels=2, bottleneck=None, pretransform=None,
                 in_channels=None, out_channels=None, soft_clip=False):
        super().__init__()
        self.downsampling_ratio = downsampling_ratio
        self.sample_rate = sample_rate
        self.latent_dim = latent_dim
        self.io_channels = io_channels
        self.in_channels = in_channels or io_channels
        self.out_channels = out_channels or io_channels
        self.min_length = downsampling_ratio
        self.bottleneck = bottleneck
        self.encoder = encoder
        self.decoder = decoder
        self.pretransform = pretransform
        self.soft_clip = soft_clip
        self.is_discrete = bottleneck is not None and bottleneck.is_discrete

    def encode(self, audio, return_info=False, skip_pretransform=False, iterate_batch=False, **kwargs):
        if self.pretransform is not None and not skip_pretransform:
            ctx = torch.enable_grad() if self.pretransform.enable_grad else torch.no_grad()
            with ctx:
                audio = _apply_fn(self.pretransform.encode, audio, iterate_batch)

        if self.encoder is not None:
            latents = _apply_fn(self.encoder, audio, iterate_batch)
        else:
            latents = audio

        if self.bottleneck is not None:
            latents, bottleneck_info = self.bottleneck.encode(latents, return_info=True, **kwargs)
            if return_info:
                return latents, bottleneck_info

        if return_info:
            return latents, {}
        return latents

    def decode(self, latents, iterate_batch=False, **kwargs):
        if self.bottleneck is not None:
            latents = _apply_fn(self.bottleneck.decode, latents, iterate_batch)

        if iterate_batch:
            decoded = _apply_fn(self.decoder, latents, True)
        else:
            decoded = self.decoder(latents, **kwargs)

        if self.pretransform is not None:
            ctx = torch.enable_grad() if self.pretransform.enable_grad else torch.no_grad()
            with ctx:
                decoded = _apply_fn(self.pretransform.decode, decoded, iterate_batch)

        if self.soft_clip:
            decoded = torch.tanh(decoded)
        return decoded

    def encode_audio(self, audio, chunked=False, overlap=32, chunk_size=128, **kwargs):
        """Encode audio to latents, optionally in overlapping chunks to save memory."""
        if not chunked:
            return self.encode(audio, **kwargs)

        samples_per_latent = self.downsampling_ratio
        total_size = audio.shape[2]
        batch_size = audio.shape[0]
        chunk_size_samples = chunk_size * samples_per_latent
        overlap_samples = overlap * samples_per_latent
        hop_size = chunk_size_samples - overlap_samples

        chunks = [audio[:, :, i:i + chunk_size_samples]
                  for i in range(0, total_size - chunk_size_samples + 1, hop_size)]
        if not chunks or (len(chunks) - 1) * hop_size + chunk_size_samples < total_size:
            chunks.append(audio[:, :, -chunk_size_samples:])

        num_chunks = len(chunks)
        y_size = total_size // samples_per_latent
        y_final = torch.zeros((batch_size, self.latent_dim, y_size), device=audio.device)
        ol = overlap // 2

        for i, chunk in enumerate(chunks):
            y_chunk = self.encode(chunk)
            if i == num_chunks - 1:
                t_end = y_size
                t_start = t_end - y_chunk.shape[2]
            else:
                t_start = i * (chunk_size - overlap)
                t_end = t_start + chunk_size

            chunk_start, chunk_end = 0, y_chunk.shape[2]
            if i > 0:
                t_start += ol
                chunk_start += ol
            if i < num_chunks - 1:
                t_end -= ol
                chunk_end -= ol
            y_final[:, :, t_start:t_end] = y_chunk[:, :, chunk_start:chunk_end]

        return y_final

    def decode_audio(self, latents, chunked=False, overlap=32, chunk_size=128, **kwargs):
        """Decode latents to audio, optionally in overlapping chunks to save memory."""
        if not chunked:
            return self.decode(latents, **kwargs)

        samples_per_latent = self.downsampling_ratio
        total_size = latents.shape[2]
        batch_size = latents.shape[0]
        hop_size = chunk_size - overlap

        chunks = [latents[:, :, i:i + chunk_size]
                  for i in range(0, total_size - chunk_size + 1, hop_size)]
        if not chunks or (len(chunks) - 1) * hop_size + chunk_size < total_size:
            chunks.append(latents[:, :, -chunk_size:])

        num_chunks = len(chunks)
        y_size = total_size * samples_per_latent
        y_final = torch.zeros((batch_size, self.out_channels, y_size), device=latents.device)
        ol = (overlap // 2) * samples_per_latent

        for i, chunk in enumerate(chunks):
            y_chunk = self.decode(chunk)
            if i == num_chunks - 1:
                t_end = y_size
                t_start = t_end - y_chunk.shape[2]
            else:
                t_start = i * hop_size * samples_per_latent
                t_end = t_start + chunk_size * samples_per_latent

            chunk_start, chunk_end = 0, y_chunk.shape[2]
            if i > 0:
                t_start += ol
                chunk_start += ol
            if i < num_chunks - 1:
                t_end -= ol
                chunk_end -= ol
            y_final[:, :, t_start:t_end] = y_chunk[:, :, chunk_start:chunk_end]

        return y_final


# --- Factory ---

def create_autoencoder_from_config(config: Dict[str, Any]):
    ae_config = config["model"]

    encoder = OobleckEncoder(**ae_config["encoder"]["config"])
    if not ae_config["encoder"].get("requires_grad", True):
        encoder.requires_grad_(False)

    decoder = OobleckDecoder(**ae_config["decoder"]["config"])
    if not ae_config["decoder"].get("requires_grad", True):
        decoder.requires_grad_(False)

    bottleneck = VAEBottleneck() if ae_config.get("bottleneck") else None

    pretransform = None
    if ae_config.get("pretransform") is not None:
        pt_cfg = ae_config["pretransform"]
        inner_ae = create_autoencoder_from_config({"sample_rate": config["sample_rate"], "model": pt_cfg["config"]})
        pretransform = AutoencoderPretransform(
            inner_ae,
            scale=pt_cfg.get("scale", 1.0),
            model_half=pt_cfg.get("model_half", False),
            iterate_batch=pt_cfg.get("iterate_batch", False),
        )
        pretransform.enable_grad = pt_cfg.get("enable_grad", False)
        pretransform.eval().requires_grad_(pretransform.enable_grad)

    return AudioAutoencoder(
        encoder, decoder,
        io_channels=ae_config["io_channels"],
        latent_dim=ae_config["latent_dim"],
        downsampling_ratio=ae_config["downsampling_ratio"],
        sample_rate=config["sample_rate"],
        bottleneck=bottleneck,
        pretransform=pretransform,
        in_channels=ae_config.get("in_channels"),
        out_channels=ae_config.get("out_channels"),
        soft_clip=ae_config["decoder"].get("soft_clip", False),
    )


# --- Discriminator ---

def get_hinge_losses(score_real, score_fake):
    gen_loss = -score_fake.mean()
    dis_loss = torch.relu(1 - score_real).mean() + torch.relu(1 + score_fake).mean()
    return dis_loss, gen_loss


class EncodecDiscriminator(nn.Module):

    def __init__(self, *args, **kwargs):
        super().__init__()
        from encodec.msstftd import MultiScaleSTFTDiscriminator
        self.discriminators = MultiScaleSTFTDiscriminator(*args, **kwargs)

    def forward(self, x):
        logits, features = self.discriminators(x)
        return logits, features

    def loss(self, x, y):
        feature_matching_distance = 0.
        logits_true, feature_true = self.forward(x)
        logits_fake, feature_fake = self.forward(y)

        dis_loss = torch.tensor(0.)
        adv_loss = torch.tensor(0.)

        for i, (scale_true, scale_fake) in enumerate(zip(feature_true, feature_fake)):
            feature_matching_distance = feature_matching_distance + sum(
                map(lambda a, b: abs(a - b).mean(), scale_true, scale_fake)
            ) / len(scale_true)
            _dis, _adv = get_hinge_losses(logits_true[i], logits_fake[i])
            dis_loss = dis_loss + _dis
            adv_loss = adv_loss + _adv

        return dis_loss, adv_loss, feature_matching_distance
