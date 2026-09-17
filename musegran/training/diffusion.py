"""Training wrapper for conditional latent diffusion with SegAlign and demo generation."""

import copy
import gc
import json
import logging
import os
import random
import typing as tp
from time import time

import pytorch_lightning as pl
import torch
import torchaudio
import wandb
from aeiou.viz import audio_spectrogram_image
from einops import rearrange
from ema_pytorch import EMA
from safetensors.torch import save_file
from torch.nn import functional as F

logger = logging.getLogger(__name__)

from ..data.load_dataset import load_dataset
from ..inference.sampling import get_alphas_sigmas, sample_discrete_euler, sample_k
from ..models.diffusion import ConditionedDiffusionModelWrapper
from .builder import create_optimizer_from_config, create_scheduler_from_config
from .losses import MSELoss, MultiLoss
from .segalign import ExternalAlignmentEncoder


def _comp_average_dic(one_path_res, skip_zero_key=None):
    """Compute per-key average across a list of score dicts.

    If skip_zero_key is specified, samples where that key equals 0 are excluded.
    """
    if not one_path_res:
        return {}

    keys = one_path_res[0].keys()
    sum_dic = {k: 0.0 for k in keys}
    valid_count = 0
    for res in one_path_res:
        if skip_zero_key and res.get(skip_zero_key) == 0.0:
            continue
        for k in keys:
            sum_dic[k] += res[k]
        valid_count += 1

    if valid_count == 0:
        return {k: 0.0 for k in keys}
    return {k: sum_dic[k] / valid_count for k in keys}


class DiffusionCondTrainingWrapper(pl.LightningModule):
    """PyTorch Lightning wrapper for conditional audio diffusion training with SegAlign."""
    def __init__(
            self,
            model: ConditionedDiffusionModelWrapper,
            sample_rate: int = 44100,
            lr: float = None,
            mask_padding: bool = False,
            mask_padding_dropout: float = 0.0,
            use_ema: bool = True,
            log_loss_info: bool = False,
            optimizer_configs: dict = None,
            cfg_dropout_prob = 0.1,
            timestep_sampler: tp.Literal["uniform", "logit_normal"] = "uniform",
            use_segalign = False,
            segalign_names = ["mert"],
            segalign_depths = [],
            segalign_coeff = 1.0
    ):
        super().__init__()
        self.strict_loading = False

        self.sample_rate = sample_rate
        self.downsampling_ratio = model.pretransform.downsampling_ratio if model.pretransform else 2048
        self.diffusion = model

        if use_ema:
            self.diffusion_ema = EMA(
                self.diffusion.model,
                beta=0.9999,
                power=3/4,
                update_every=1,
                update_after_step=1,
                include_online_model=False
            )
        else:
            self.diffusion_ema = None

        self.mask_padding = mask_padding
        self.mask_padding_dropout = mask_padding_dropout

        self.cfg_dropout_prob = cfg_dropout_prob

        self.rng = torch.quasirandom.SobolEngine(1, scramble=True)

        self.timestep_sampler = timestep_sampler

        self.diffusion_objective = model.diffusion_objective

        self.use_segalign = use_segalign
        if self.use_segalign and hasattr(model, 'conditioner') and model.conditioner is not None:
            need_embeds = any(n in segalign_names for n in ("glap", "condss"))
            for module in model.conditioner.conditioners.values():
                if hasattr(module, 'update_metadata'):
                    module.update_metadata = True
                    module.collect_embeds = need_embeds
        if self.use_segalign:
            self.segalign_names = segalign_names
            self.segalign_depths = segalign_depths
            self.segalign_coeff = segalign_coeff
            self.ea_encoder = ExternalAlignmentEncoder(
                external_alignment_names=self.segalign_names, sample_rate=self.sample_rate
            )
            self.cosine_loss = torch.nn.CosineEmbeddingLoss(margin=0.0, reduction="mean")

            if "glap" in segalign_names:
                from musegran.models.conditioner_utils.glap import GlapBaseBert
                self.glap_resampler = torchaudio.transforms.Resample(orig_freq=self.sample_rate, new_freq=16000)
                self.glap_audio_encoder = GlapBaseBert().eval()
                self.glap_audio_encoder.requires_grad_(False)

        self.loss_modules = [
            MSELoss("output",
                   "targets",
                   weight=1.0,
                   mask_key="padding_mask" if self.mask_padding else None,
                   name="mse_loss"
            )
        ]

        self.losses = MultiLoss(self.loss_modules)

        self.log_loss_info = log_loss_info

        assert lr is not None or optimizer_configs is not None, "Must specify either lr or optimizer_configs in training config"

        if optimizer_configs is None:
            optimizer_configs = {
                "diffusion": {
                    "optimizer": {
                        "type": "Adam",
                        "config": {
                            "lr": lr
                        }
                    }
                }
            }
        else:
            if lr is not None:
                logger.warning("learning_rate and optimizer_configs both specified in config. Ignoring learning_rate and using optimizer_configs.")

        self.optimizer_configs = optimizer_configs

    def _build_per_sample_training_info(self, training_outputs, batch_size):
        """Flatten conditioner-keyed training_outputs into a per-sample list.

        When multiple conditioners provide segment_boundaries at different compress_ratios,
        we keep all of them keyed by compress_ratio so each loss can pick the right one.
        """
        per_sample = [{} for _ in range(batch_size)]

        for _key, info in training_outputs.items():
            if 'segment_embeds' in info and info['segment_embeds']:
                for i, se in enumerate(info['segment_embeds']):
                    per_sample[i]['segment_embeds'] = se
            if 'segment_boundaries' in info and info['segment_boundaries']:
                for i, sb in enumerate(info['segment_boundaries']):
                    per_sample[i]['segment_boundaries'] = sb

        return per_sample

    def _compute_segalign_loss(self, segalign_projections, ea_hidden_states, latent_seq_len, audios, metadata, training_outputs):
        """Compute SegAlign loss: External Alignment (EA) + Condition Self-Supervision (CondSS)."""
        bs = segalign_projections[0].shape[0]
        per_sample_info = self._build_per_sample_training_info(training_outputs, bs)

        losses = []
        for projection, hidden_state, name in zip(
            segalign_projections, ea_hidden_states, self.segalign_names
        ):
            ea_loss = 0.0
            ea_count = 0
            condss_loss = 0.0
            condss_count = 0

            if hidden_state is None:
                for i, z_tilde in enumerate(projection):
                    z_tilde = z_tilde[-latent_seq_len:]
                    info_i = per_sample_info[i] if per_sample_info else None
                    if name == "glap":
                        v = self._ea_segment_loss(z_tilde, audios[i], metadata[i], info_i)
                        if v is not None:
                            ea_loss += v
                            ea_count += 1
                    elif name == "condss":
                        v = self._condss_loss(z_tilde, info_i)
                        if v is not None:
                            condss_loss += v
                            condss_count += 1
            else:
                for i, (z, z_tilde) in enumerate(zip(hidden_state, projection)):
                    z_tilde = z_tilde[-latent_seq_len:]
                    info_i = per_sample_info[i] if per_sample_info else None

                    wav_sec = min(metadata[i]['crop_end'] - metadata[i]['crop_start'], 180)
                    z_tilde_latent_length = round(wav_sec * self.sample_rate / self.downsampling_ratio)
                    scale_ratio = z.shape[0] / z_tilde_latent_length

                    z_tilde = z_tilde[:z_tilde_latent_length]
                    z_tilde = F.interpolate(
                        z_tilde.unsqueeze(0).transpose(1, 2),
                        size=len(z), mode="linear", align_corners=False,
                    ).transpose(1, 2).squeeze(0)

                    z_tilde = F.normalize(z_tilde, dim=-1)
                    z = F.normalize(z, dim=-1)
                    v = self._ea_loss(z, z_tilde, info_i, scale_ratio=scale_ratio)
                    if v is not None:
                        ea_loss += v
                        ea_count += 1

            if ea_count > 0:
                losses.append((name, ea_loss / ea_count))
            if condss_count > 0:
                losses.append(("condss", condss_loss / condss_count))

        if not losses:
            return torch.tensor(0.0, device=segalign_projections[0].device), []
        total_loss = sum([l[-1] for l in losses]) / len(losses)
        return total_loss * self.segalign_coeff, losses

    def _ea_loss(self, z, z_tilde, training_info=None, scale_ratio=1.0, alpha=0.9):
        """External Alignment: segment-level + global cosine loss against frozen encoder."""
        max_len = z.shape[0]
        target = torch.ones(max_len, device=z.device)

        global_loss = self.cosine_loss(z, z_tilde, target)
        if training_info is None or 'segment_boundaries' not in training_info:
            return global_loss

        audio_boundaries = training_info['segment_boundaries']
        if not audio_boundaries:
            return global_loss

        latent_fps = self.sample_rate / self.downsampling_ratio
        tmp_loss = 0.0
        valid = 0
        for start_sec, end_sec in audio_boundaries:
            start = min(int(start_sec * latent_fps * scale_ratio), max_len)
            end = min(int(end_sec * latent_fps * scale_ratio), max_len)
            if start >= end:
                continue
            tmp_loss += 1 - F.cosine_similarity(z[start:end], z_tilde[start:end], dim=0).mean()
            valid += 1

        if valid == 0:
            return global_loss
        loss = tmp_loss / valid
        loss = loss * alpha + global_loss * (1 - alpha)
        return loss

    def _condss_loss(self, z_tilde, training_info=None):
        """Condition Self-Supervision: enforce segment embeddings predict masked segment DiT states."""
        max_len = z_tilde.shape[0]

        if training_info is None or 'segment_boundaries' not in training_info:
            return None

        audio_boundaries = training_info['segment_boundaries']
        cond_emb, mask_idx = training_info['segment_embeds']
        if len(audio_boundaries) == 0 or len(audio_boundaries) == len(mask_idx):
            return None

        latent_fps = self.sample_rate / self.downsampling_ratio
        tmp_loss = 0.0
        valid_count = 0
        for i, (start_sec, end_sec) in enumerate(audio_boundaries):
            start = min(int(start_sec * latent_fps), max_len)
            end = min(int(end_sec * latent_fps), max_len)
            if i in mask_idx or start >= end:
                continue
            tmp_loss += 1 - F.cosine_similarity(z_tilde[start:end].mean(0), cond_emb[i].squeeze(0).clone(), dim=0)
            valid_count += 1
        if valid_count == 0:
            return None
        return tmp_loss / valid_count

    def _ea_segment_loss(self, z_tilde, audio, metadata=None, training_info=None):
        """External Alignment (segment-level): mean-pool DiT states per segment, align with audio encoder embedding."""
        max_len = z_tilde.shape[0]

        if training_info is None or 'segment_boundaries' not in training_info:
            return None

        audio_boundaries = training_info['segment_boundaries']
        _, mask_idx = training_info['segment_embeds']
        if len(audio_boundaries) == 0 or len(audio_boundaries) == len(mask_idx):
            return None

        if metadata is None or metadata.get('structure') is None:
            return None

        structure_data = metadata['structure']
        if isinstance(structure_data, dict):
            structure_infos = structure_data['structure_infos']
        else:
            structure_infos = structure_data[0]

        if isinstance(structure_infos, str):
            return None

        latent_fps = self.sample_rate / self.downsampling_ratio
        mono_16k = self.glap_resampler(audio.mean(dim=0, keepdim=True))

        # Collect valid segments for batched GLAP encoding
        audio_slices = []
        dit_embs = []
        for i, (start_sec, end_sec) in enumerate(audio_boundaries):
            if i in mask_idx:
                continue

            lat_start = min(int(start_sec * latent_fps), max_len)
            lat_end = min(int(end_sec * latent_fps), max_len)
            if lat_start >= lat_end:
                continue

            audio_start = int(start_sec * 16000)
            audio_end = min(int(end_sec * 16000), mono_16k.shape[1])

            audio_slice = mono_16k[:, audio_start:audio_end]
            if audio_slice.shape[1] < 16000:
                continue

            audio_slices.append(audio_slice.squeeze(0))
            dit_embs.append(z_tilde[lat_start:lat_end].mean(0))

        if len(audio_slices) == 0:
            return None

        # Batch encode: pad slices to max length, encode in one pass
        audio_lengths = torch.tensor([s.shape[0] for s in audio_slices], device=audio.device)
        max_audio_len = audio_lengths.max().item()
        batched_audio = torch.zeros(len(audio_slices), max_audio_len, device=audio.device)
        for idx, s in enumerate(audio_slices):
            batched_audio[idx, :s.shape[0]] = s

        with torch.no_grad():
            glap_embs = self.glap_audio_encoder.glap.encode_audio(batched_audio, audio_lengths)

        dit_embs = torch.stack(dit_embs)
        dit_embs = F.normalize(dit_embs, dim=-1)
        glap_embs = F.normalize(glap_embs, dim=-1)

        losses = 1 - F.cosine_similarity(dit_embs, glap_embs, dim=-1)
        return losses.mean()

    def on_save_checkpoint(self, checkpoint):
        # Only keep trainable parameters — frozen VAE/conditioner backbones are loaded
        # separately at inference time, no need to bloat the checkpoint.
        trainable_keys = {n for n, p in self.named_parameters() if p.requires_grad}
        checkpoint["state_dict"] = {
            k: v for k, v in checkpoint["state_dict"].items() if k in trainable_keys
        }

    def configure_optimizers(self):
        logger.info("Trainable Params: %s",
            [n for n, p in self.diffusion.named_parameters() if p.requires_grad])

        diffusion_opt_config = self.optimizer_configs['diffusion']
        conditioner_opt_config = self.optimizer_configs.get('conditioner', None)

        if conditioner_opt_config is not None:
            conditioner_param_ids = set(
                id(p) for p in self.diffusion.conditioner.parameters()
            )

            conditioner_params = [
                p for p in self.diffusion.conditioner.parameters() if p.requires_grad
            ]
            diffusion_params = [
                p for p in self.diffusion.parameters()
                if p.requires_grad and id(p) not in conditioner_param_ids
            ]

            logger.info("Diffusion params: %d", sum(p.numel() for p in diffusion_params))
            logger.info("Conditioner params: %d", sum(p.numel() for p in conditioner_params))

            diff_opt_conf = diffusion_opt_config['optimizer']['config']
            cond_opt_conf = conditioner_opt_config['optimizer']['config']

            from torch.optim import AdamW
            opt = AdamW([
                {
                    "params": diffusion_params,
                    "lr": diff_opt_conf['lr'],
                    "betas": tuple(diff_opt_conf.get('betas', [0.9, 0.999])),
                    "weight_decay": diff_opt_conf.get('weight_decay', 1e-3)
                },
                {
                    "params": conditioner_params,
                    "lr": cond_opt_conf['lr'],
                    "betas": tuple(cond_opt_conf.get('betas', [0.9, 0.999])),
                    "weight_decay": cond_opt_conf.get('weight_decay', 1e-2)
                },
            ])

            if "scheduler" in diffusion_opt_config:
                sched = create_scheduler_from_config(
                    diffusion_opt_config['scheduler'], opt
                )
                return [opt], [{"scheduler": sched, "interval": "step"}]

            return [opt]

        else:
            params_to_train = [
                p for p in self.diffusion.parameters() if p.requires_grad
            ]

            opt_diff = create_optimizer_from_config(
                diffusion_opt_config['optimizer'], params_to_train
            )

            if "scheduler" in diffusion_opt_config:
                sched_diff = create_scheduler_from_config(
                    diffusion_opt_config['scheduler'], opt_diff
                )
                return [opt_diff], [{"scheduler": sched_diff, "interval": "step"}]

            return [opt_diff]

    def training_step(self, batch, batch_idx):
        """Single training step: encode, noise, denoise, compute diffusion + SegAlign losses."""
        latents, audios, metadata = batch

        pre_encoded = metadata[0].get("pre_encoded", False)
        reals = latents if pre_encoded else audios

        if reals.ndim == 4 and reals.shape[0] == 1:
            reals = reals[0]

        loss_info = {}

        diffusion_input = reals

        ea_hidden_states = None
        if self.use_segalign:
            ea_hidden_states = self.ea_encoder(audios, metadata, train=self.training)


        with torch.amp.autocast('cuda'):
            conditioning, training_outputs = self.diffusion.conditioner(metadata, self.device)

        use_padding_mask = self.mask_padding and random.random() > self.mask_padding_dropout

        if use_padding_mask:
            padding_masks = torch.stack([md["padding_mask"] for md in metadata], dim=0).to(self.device)

        if self.diffusion.pretransform is not None:
            self.diffusion.pretransform.to(self.device)

            if not pre_encoded:
                with torch.amp.autocast('cuda') and torch.set_grad_enabled(self.diffusion.pretransform.enable_grad):
                    self.diffusion.pretransform.train(self.diffusion.pretransform.enable_grad)
                    diffusion_input = self.diffusion.pretransform.encode(diffusion_input)
            else:
                if hasattr(self.diffusion.pretransform, "scale") and self.diffusion.pretransform.scale != 1.0:
                    diffusion_input = diffusion_input / self.diffusion.pretransform.scale

            if use_padding_mask:
                padding_masks = F.interpolate(padding_masks.unsqueeze(1).float(), size=diffusion_input.shape[2], mode="nearest").squeeze(1).bool()

        if self.timestep_sampler == "uniform":
            t = self.rng.draw(reals.shape[0])[:, 0].to(self.device)
        elif self.timestep_sampler == "logit_normal":
            t = torch.sigmoid(torch.randn(reals.shape[0], device=self.device))

        if self.diffusion_objective == "v":
            alphas, sigmas = get_alphas_sigmas(t)
        elif self.diffusion_objective == "rectified_flow":
            alphas, sigmas = 1-t, t

        alphas = alphas[:, None, None]
        sigmas = sigmas[:, None, None]
        noise = torch.randn_like(diffusion_input)
        noised_inputs = diffusion_input * alphas + noise * sigmas

        if self.diffusion_objective == "v":
            targets = noise * alphas - diffusion_input * sigmas
        elif self.diffusion_objective == "rectified_flow":
            targets = noise - diffusion_input

        extra_args = {}

        if use_padding_mask:
            extra_args["mask"] = padding_masks

        if self.use_segalign:
            extra_args["segalign_depths"] = self.segalign_depths

        with torch.amp.autocast('cuda'):
            output = self.diffusion(
                                    noised_inputs,
                                    t,
                                    cond=conditioning,
                                    cfg_dropout_prob=self.cfg_dropout_prob,
                                    **extra_args
                                    )

            if self.use_segalign:
                output, segalign_projections = output

            loss_info.update({
                "output": output,
                "targets": targets,
                "padding_mask": padding_masks if use_padding_mask else None,
            })

            loss, losses = self.losses(loss_info)

            if self.log_loss_info:
                num_loss_buckets = 10
                bucket_size = 1 / num_loss_buckets
                loss_all = F.mse_loss(output, targets, reduction="none")

                sigmas = rearrange(self.all_gather(sigmas), "w b c n -> (w b) c n").squeeze()
                loss_all = rearrange(self.all_gather(loss_all), "w b c n -> (w b) c n")
                loss_all = torch.stack([loss_all[(sigmas >= i) & (sigmas < i + bucket_size)].mean() for i in torch.arange(0, 1, bucket_size).to(self.device)])

                debug_log_dict = {
                    f"model/loss_all_{i/num_loss_buckets:.1f}": loss_all[i].detach() for i in range(num_loss_buckets) if not torch.isnan(loss_all[i])
                }

                self.log_dict(debug_log_dict)

        if self.use_segalign:
            segalign_loss, segalign_losses = self._compute_segalign_loss(
                segalign_projections, ea_hidden_states, latents.shape[-1], audios, metadata, training_outputs
            )
            loss = loss + segalign_loss

        log_dict = {
            'train/loss': loss.detach(),
            'train/std_data': diffusion_input.std(),
            'train/lr_diffusion': self.trainer.optimizers[0].param_groups[0]['lr'],
        }

        if len(self.trainer.optimizers[0].param_groups) > 1:
            log_dict['train/lr_conditioner'] = self.trainer.optimizers[0].param_groups[1]['lr']

        for loss_name, loss_value in losses.items():
            log_dict[f"train/{loss_name}"] = loss_value.detach()

        if self.use_segalign:
            for l in segalign_losses:
                log_dict[f"train/{l[0]}"] = l[1].detach()

        self.log_dict(log_dict, prog_bar=True, on_step=True)

        return loss

    def on_before_zero_grad(self, *args, **kwargs):
        """Update EMA weights after each optimizer step."""
        if self.diffusion_ema is not None:
            self.diffusion_ema.update()

    def export_model(self, path: str, use_safetensors: bool = False):
        """Export model weights (using EMA if available) to disk."""
        if self.diffusion_ema is not None:
            self.diffusion.model = self.diffusion_ema.ema_model

        if use_safetensors:
            save_file(self.diffusion.state_dict(), path)
        else:
            torch.save({"state_dict": self.diffusion.state_dict()}, path)

    def export_trainable_weights(self, path: str):
        """Export only trainable parameters (DiT + conditioner adapters) for inference."""
        trainable_names = {n for n, p in self.diffusion.named_parameters() if p.requires_grad}

        if self.diffusion_ema is not None:
            # Temporarily swap EMA weights into the model, export, then swap back
            orig_state = {k: v.clone() for k, v in self.diffusion.model.state_dict().items()}
            self.diffusion.model.load_state_dict(self.diffusion_ema.ema_model.state_dict())
            state_dict = {k: v for k, v in self.diffusion.state_dict().items() if k in trainable_names}
            self.diffusion.model.load_state_dict(orig_state)
        else:
            state_dict = {k: v for k, v in self.diffusion.state_dict().items() if k in trainable_names}

        torch.save({"state_dict": state_dict}, path)


class DiffusionCondDemoCallback(pl.Callback):
    """Lightning callback that generates and scores demo audio at regular training intervals."""

    def __init__(self,
                 demo_every=2000,
                 num_demos=8,
                 sample_size=65536,
                 demo_steps=250,
                 sample_rate=48000,
                 demo_conditioning: tp.Optional[tp.Dict[str, tp.Any]] = {},
                 demo_cfg_scales: tp.Optional[tp.List[int]] = [3, 5, 7],
                 demo_cond_from_batch: bool = False,
                 display_audio_cond: bool = False,
                 demo_negative_cond: tp.Union[bool, str] = False,
                 generate_demo: bool = True,
                 experiment_id: str = None,
                 dataset_func_args: tp.Optional[dict] = {},
                 valid_metrics: tp.Optional[tp.List[str]] = ["audiobox", "songeval"],
    ):
        super().__init__()

        self.demo_every = demo_every
        self.demo_samples = sample_size
        self.demo_steps = demo_steps
        self.sample_rate = sample_rate
        self.last_demo_step = -1
        self.demo_cfg_scales = demo_cfg_scales
        self.demo_conditioning = demo_conditioning

        self.demo_metrics = {}
        self.current_demo_step = -1

        if isinstance(self.demo_conditioning, str):
            file_ext = os.path.splitext(self.demo_conditioning)[1].lower()
            with open(self.demo_conditioning, "r", encoding="utf-8") as f:
                if file_ext == '.jsonl':
                    self.demo_conditioning = []
                    for line_num, line in enumerate(f, 1):
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            json_obj = json.loads(line)
                            self.demo_conditioning.append(json_obj)
                        except json.JSONDecodeError as e:
                            raise ValueError(f"JSONL parse error at line {line_num}: {str(e)}")
                else:
                    self.demo_conditioning = json.load(f)

        self.num_demos = len(self.demo_conditioning)
        for i in range(self.num_demos):
            if isinstance(self.demo_conditioning[i], str):
                self.demo_conditioning[i] = self._get_democond_from_json(self.demo_conditioning[i])
            elif isinstance(self.demo_conditioning[i], dict):
                self.demo_conditioning[i] = load_dataset(self.demo_conditioning[i], None, **dataset_func_args)

        self.generate_demo = generate_demo
        self.demo_negative_cond = demo_negative_cond
        self.valid_metrics = valid_metrics
        self.conver_mp3 = True
        self.is_batch_inference = True
        self.experiment_id = experiment_id

        from datetime import datetime
        self._run_timestamp = datetime.now().strftime("%y-%m-%d_%H-%M-%S")

        self.demo_cond_from_batch = demo_cond_from_batch
        self.display_audio_cond = display_audio_cond
        self._scorers = {}
        self._scorer_device = None

    @staticmethod
    def _get_democond_from_json(json_file):
        """Build demo conditioning dict from a structure annotation JSON file."""
        with open(json_file, "r", encoding="utf-8") as f:
            json_infos = json.load(f)

        json_audio_fn = list(json_infos.keys())[0]
        json_labels = list(json_infos.values())[0]

        json_genre = json_labels['global analysis']['genre_mtg_jamendo_87']
        json_prompt = json_labels['overall_description']['summary_description'].strip()
        if isinstance(json_genre, list) and len(json_genre) > 0:
            random.shuffle(json_genre)
            json_genre_str = ', '.join(json_genre)
            json_prompt = json_prompt + ' Additional_Genre:' + json_genre_str
        global_prompt = json_prompt

        bpm = float(json_labels['global analysis']['bpm'])
        seconds_total = float(json_labels['basic_info']['duration_seconds'])
        return {
            'prompt': [json_file, [0, seconds_total], json_audio_fn, global_prompt],
            'chord': None,
            'beat': {"bpm": bpm, "seconds_start": 0, "seconds_total": seconds_total},
            'seconds_start': 0,
            'seconds_total': int(seconds_total),
            'music_key': json_labels['global analysis']['key'],
            'structure': [json_file, [0, seconds_total], None]
        }

    def _cfg_label(self, cfg_scale):
        """Convert a cfg_scale value to a string label for logging."""
        if isinstance(cfg_scale, list):
            return ''.join(map(str, cfg_scale))
        return str(cfg_scale)

    def _run_scorer(self, all_cfg_input_infos, cfg_labels, scorer_name, scorer_fn,
                    metric_keys, skip_zero_key=None, post_process=None):
        """Run a scorer over all cfg paths, collect averages and metrics.

        Args:
            all_cfg_input_infos: list of input_info lists, one per cfg_scale
            cfg_labels: list of string labels corresponding to each cfg_scale
            scorer_name: name used in metric keys and print messages
            scorer_fn: callable(input_infos) -> list of score dicts
            metric_keys: which keys from avg dict to log to demo_metrics
            skip_zero_key: passed to _comp_average_dic
            post_process: optional callable(scores, avg) for extra processing
        """
        logger.info("== The %s scoring begins ==", scorer_name)
        results = []
        for idx, input_infos in enumerate(all_cfg_input_infos):
            scores = scorer_fn(input_infos)

            if post_process:
                post_process(scores)

            avg = _comp_average_dic(scores, skip_zero_key=skip_zero_key)
            scores.append(avg)

            cfg_key = f"Demo/cfg_{cfg_labels[idx]}/{scorer_name}"
            for key in metric_keys:
                self.demo_metrics[f"{cfg_key}_{key}"] = avg.get(key, 0.0)

            summary = ", ".join(f"{k}: {avg.get(k, 0):.2f}" for k in metric_keys if isinstance(avg.get(k, 0), float))
            logger.info("  [%s] %s --- %s", scorer_name, os.path.basename(input_infos[0]['path']), summary)

            results.append(scores)
        logger.info("== The %s scoring is over ==", scorer_name)
        return results

    @torch.no_grad()
    def on_train_batch_end(self, trainer, module: DiffusionCondTrainingWrapper, outputs, batch, batch_idx):
        if not self.generate_demo:
            return

        if (trainer.global_step - 1) % self.demo_every != 0 or self.last_demo_step == trainer.global_step:
            return

        self.last_demo_step = trainer.global_step

        if trainer.global_rank == 0:
            gc.collect()
            torch.cuda.empty_cache()

            module.eval()
            logger.info("Generating demo")

            self.demo_metrics.clear()

            demo_samples = self.demo_samples

            demo_cond = self.demo_conditioning

            if self.demo_cond_from_batch:
                demo_cond = batch[-1][:self.num_demos]

            # Build negative conditions
            if self.demo_negative_cond:
                if isinstance(self.demo_negative_cond, str):
                    negative_prompt = self.demo_negative_cond
                else:
                    negative_prompt = 'sampled vocals, noise, distorted harmonics, artifacts, low quality'

                demo_negative_cond = copy.deepcopy(demo_cond)
                for i in range(self.num_demos):
                    for key in demo_negative_cond[i].keys():
                        if key == 'prompt':
                            demo_negative_cond[i][key] = negative_prompt
                        elif key in ('seconds_start', 'seconds_total'):
                            continue
                        else:
                            demo_negative_cond[i][key] = None
            else:
                demo_negative_cond = None

            if module.diffusion.pretransform is not None:
                demo_samples = demo_samples // module.diffusion.pretransform.downsampling_ratio

            noise = torch.randn([self.num_demos, module.diffusion.io_channels, demo_samples]).to(module.device)

            try:
                logger.info("Getting conditioning")
                with torch.inference_mode(), torch.amp.autocast('cuda'):
                    if self.is_batch_inference:
                        conditioning, _ = module.diffusion.conditioner(demo_cond, module.device)
                        cond_inputs = module.diffusion.get_conditioning_inputs(conditioning)

                        if demo_negative_cond is not None:
                            neg_conditioning, _ = module.diffusion.conditioner(demo_negative_cond, module.device)
                            neg_cond_inputs = module.diffusion.get_conditioning_inputs(neg_conditioning, negative=True)
                        else:
                            neg_cond_inputs = {}
                    else:
                        conditioning = [module.diffusion.conditioner([demo_cond[i]], module.device)[0] for i in range(self.num_demos)]
                        cond_inputs = [module.diffusion.get_conditioning_inputs(conditioning[i]) for i in range(self.num_demos)]

                        if demo_negative_cond is not None:
                            neg_conditioning = [module.diffusion.conditioner([demo_negative_cond[i]], module.device)[0] for i in range(self.num_demos)]
                            neg_cond_inputs = [module.diffusion.get_conditioning_inputs(neg_conditioning[i], negative=True) for i in range(self.num_demos)]
                        else:
                            neg_cond_inputs = [{} for _ in range(self.num_demos)]

                log_dict = {}

                if self.display_audio_cond:
                    audio_inputs = torch.cat([cond["audio"] for cond in demo_cond], dim=0)
                    audio_inputs = rearrange(audio_inputs, 'b d n -> d (b n)')

                    filename = f'demo_audio_cond_{trainer.global_step:08}.wav'
                    audio_inputs = audio_inputs.to(torch.float32).mul(32767).to(torch.int16).cpu()
                    torchaudio.save(filename, audio_inputs, self.sample_rate)
                    log_dict['demo_audio_cond'] = wandb.Audio(filename, sample_rate=self.sample_rate, caption="Audio conditioning")
                    log_dict["demo_audio_cond_melspec_left"] = wandb.Image(audio_spectrogram_image(audio_inputs))
                    trainer.logger.experiment.log(log_dict)

                demo_audio_paths = []
                for cfg_scale in self.demo_cfg_scales:

                    cfg_str = self._cfg_label(cfg_scale)
                    negative_cond_inputs = neg_cond_inputs

                    logger.info("Generating demo for cfg scale %s", cfg_str)

                    with torch.inference_mode(), torch.amp.autocast('cuda'):
                        model = module.diffusion_ema.model if module.diffusion_ema is not None else module.diffusion.model

                        if module.diffusion_objective == "v":
                            if self.is_batch_inference:
                                fakes = sample_k(model, noise, self.demo_steps, sigma_min=0.3, sigma_max=500, sampler_type="dpmpp-3m-sde", **cond_inputs, **negative_cond_inputs, cfg_scale=cfg_scale)
                            else:
                                fakes = [sample_k(model, noise[i].unsqueeze(0), self.demo_steps, sigma_min=0.3, sigma_max=500, sampler_type="dpmpp-3m-sde", **cond_inputs[i], **negative_cond_inputs[i], cfg_scale=cfg_scale) for i in range(self.num_demos)]

                        elif module.diffusion_objective == "rectified_flow":
                            if self.is_batch_inference:
                                fakes = sample_discrete_euler(model, noise, self.demo_steps, **cond_inputs, cfg_scale=cfg_scale)
                            else:
                                fakes = [sample_discrete_euler(model, noise[i].unsqueeze(0), self.demo_steps, **cond_inputs[i], **negative_cond_inputs[i], cfg_scale=cfg_scale) for i in range(self.num_demos)]

                        if module.diffusion.pretransform is not None:
                            if self.is_batch_inference:
                                fakes = module.diffusion.pretransform.decode(fakes)
                                fakes = rearrange(fakes, 'b d n -> d (b n)')
                            else:
                                decoded_patches = []
                                for i in range(self.num_demos):
                                    fake_single = fakes[i].unsqueeze(0) if len(fakes[i].shape) == 2 else fakes[i]
                                    decoded_single = module.diffusion.pretransform.decode(fake_single)
                                    decoded_patches.append(decoded_single.squeeze(0))
                                fakes = torch.cat(decoded_patches, dim=1)

                    log_dict = {}

                    experiment_id = self.experiment_id if self.experiment_id is not None else trainer.logger.experiment.id
                    save_dir = os.path.join(trainer.default_root_dir, experiment_id, "demo", self._run_timestamp)
                    os.makedirs(save_dir, exist_ok=True)
                    filename = os.path.join(save_dir, f'demo_cfg_{cfg_str}_{trainer.global_step:08}.wav')

                    fakes = fakes.div(torch.max(torch.abs(fakes))).mul(32767).to(torch.int16).cpu()
                    torchaudio.save(filename, fakes, self.sample_rate)

                    if self.conver_mp3:
                        mp3_filename = filename.replace('.wav', '.mp3')
                        torchaudio.save(mp3_filename, fakes, self.sample_rate, format="mp3")
                        os.remove(filename)
                        filename = mp3_filename

                    demo_audio_paths.append(filename)

                    log_dict[f'demo_cfg_{cfg_str}'] = wandb.Audio(filename, sample_rate=self.sample_rate, caption='Reconstructed')
                    log_dict[f'demo_melspec_left_cfg_{cfg_str}'] = wandb.Image(audio_spectrogram_image(fakes))

                    trainer.logger.experiment.log(log_dict)

                start_score_time = time()
                try:
                    self._score_demos(demo_audio_paths, module.device)
                except Exception:
                    logger.warning("Scoring failed, skipping", exc_info=True)
                logger.info("Scoring took %.2fs", time() - start_score_time)

                self.current_demo_step = trainer.global_step

                del fakes

            except Exception as e:
                raise e
            finally:
                gc.collect()
                torch.cuda.empty_cache()
                module.train()
        if torch.distributed.is_initialized():
            torch.distributed.barrier()

    def _score_demos(self, audio_paths, device):
        """Run all configured scoring metrics on generated demo audio."""
        valid_modes = {'audiobox', 'songeval'}
        mode = self.valid_metrics
        if not mode or any(m not in valid_modes for m in mode):
            return

        cfg_labels = [self._cfg_label(s) for s in self.demo_cfg_scales]

        per_demo_seconds = self.demo_samples / self.sample_rate

        all_cfg_input_infos = []
        for audio_idx, audio_path in enumerate(audio_paths):
            input_infos = []
            for i in range(self.num_demos):
                input_infos.append({
                    "path": audio_path,
                    "start_time": per_demo_seconds * i,
                    "end_time": per_demo_seconds * i + self.demo_conditioning[i]['seconds_total'],
                })
            all_cfg_input_infos.append(input_infos)

        num_cfgs = len(all_cfg_input_infos)
        all_results = {
            "audiobox": [None] * num_cfgs,
            "songeval": [None] * num_cfgs,
        }

        # Lazy-load and cache scorers to avoid re-instantiation every demo step
        if self._scorer_device != device:
            self._scorers.clear()
            self._scorer_device = device

        if "audiobox" in mode:
            if "audiobox" not in self._scorers:
                from score_tools.audiobox_aesthetics import Scorer as AudioboxScorer
                self._scorers["audiobox"] = AudioboxScorer(device=device)

            def _audiobox_post_process(scores):
                for item in scores:
                    item['maybe_noise_base_audiobox'] = item['CE'] <= 7.0 and item['CU'] <= 7.0 and item['PQ'] <= 7.0

            with torch.no_grad():
                all_results["audiobox"] = self._run_scorer(
                    all_cfg_input_infos, cfg_labels,
                    scorer_name="audiobox",
                    scorer_fn=self._scorers["audiobox"].forward,
                    metric_keys=['CE', 'CU', 'PQ'],
                    post_process=_audiobox_post_process,
                )

        if "songeval" in mode:
            if "songeval" not in self._scorers:
                from score_tools.songeval import Scorer as SongevalScorer
                self._scorers["songeval"] = SongevalScorer(device=device)

            with torch.no_grad():
                all_results["songeval"] = self._run_scorer(
                    all_cfg_input_infos, cfg_labels,
                    scorer_name="songeval",
                    scorer_fn=self._scorers["songeval"].forward,
                    metric_keys=['Coherence', 'Musicality', 'Memorability', 'Clarity'],
                )

        # Write per-path JSON results
        self._write_score_json(audio_paths, all_results, mode)

    def _write_score_json(self, audio_paths, all_results, mode):
        """Write per-path scoring results to JSON files."""
        per_demo_seconds = self.demo_samples / self.sample_rate

        for audio_idx, audio_path in enumerate(audio_paths):
            one_path_res_dic_list = []

            ab = all_results["audiobox"][audio_idx]
            se = all_results["songeval"][audio_idx]

            num_segments = 0
            for res in [ab, se]:
                if res is not None and len(res) > 0:
                    num_segments = len(res) - 1
                    break

            for segment_idx in range(num_segments):
                start_time = per_demo_seconds * segment_idx
                end_time = start_time + self.demo_conditioning[segment_idx]['seconds_total']
                res_dic = {
                    "path": audio_path,
                    "start_time": start_time,
                    "end_time": end_time,
                }
                if "audiobox" in mode and ab is not None:
                    res_dic["audiobox_score"] = ab[segment_idx]
                if "songeval" in mode and se is not None:
                    res_dic["songeval_score"] = se[segment_idx]

                one_path_res_dic_list.append(res_dic)

            avg_dic = {"path": audio_path, "Is_average": True}
            if "audiobox" in mode and ab is not None:
                avg_dic["audiobox_average_score"] = ab[-1]
            if "songeval" in mode and se is not None:
                avg_dic["songeval_average_score"] = se[-1]
            one_path_res_dic_list.append(avg_dic)

            json_path = f"{audio_path.rsplit('.', 1)[0]}.json"
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(one_path_res_dic_list, f, ensure_ascii=False, indent=2)

