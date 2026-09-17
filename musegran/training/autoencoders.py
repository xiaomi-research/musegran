"""Training wrapper for audio autoencoder (VAE) with discriminator and EMA."""

import json
import logging
import os
from time import time

import pytorch_lightning as pl
import torch
import torchaudio
import wandb
from aeiou.viz import pca_point_cloud, audio_spectrogram_image, tokens_spectrogram_image
from einops import rearrange
from ema_pytorch import EMA
from pytorch_lightning.utilities.rank_zero import rank_zero_only
from safetensors.torch import save_model

from ..models.autoencoders import AudioAutoencoder, EncodecDiscriminator, VAEBottleneck
from .builder import create_optimizer_from_config, create_scheduler_from_config
from .losses import MultiLoss, AuralossLoss, ValueLoss, L1Loss
from .losses.auraloss import SumAndDifferenceSTFTLoss, MultiResolutionSTFTLoss

logger = logging.getLogger(__name__)


def _unpack_batch(batch):
    """Unpack batch → (reals, encoder_input, is_superres).

    Super-resolution batches have 3 elements: (lossless, compressed, metadata).
    Standard batches have 2 elements: (audio, metadata).
    """
    if len(batch) == 3:
        reals, encoder_input, _ = batch
        is_superres = True
    else:
        reals, _ = batch
        encoder_input = reals
        is_superres = False

    if reals.ndim == 4 and reals.shape[0] == 1:
        reals = reals[0]
    if encoder_input is not reals and encoder_input.ndim == 4 and encoder_input.shape[0] == 1:
        encoder_input = encoder_input[0]

    return reals, encoder_input, is_superres


class AutoencoderTrainingWrapper(pl.LightningModule):
    def __init__(
            self,
            autoencoder: AudioAutoencoder,
            lr: float = 1e-4,
            warmup_steps: int = 0,
            encoder_freeze_on_warmup: bool = False,
            sample_rate=48000,
            loss_config: dict = None,
            optimizer_configs: dict = None,
            use_ema: bool = True,
            ema_copy = None,
            force_input_mono = False,
            latent_mask_ratio = 0.0,
            teacher_model: AudioAutoencoder = None
    ):
        super().__init__()

        self.automatic_optimization = False

        self.autoencoder = autoencoder

        self.warmed_up = False
        self.warmup_steps = warmup_steps
        self.encoder_freeze_on_warmup = encoder_freeze_on_warmup
        self.lr = lr

        self.force_input_mono = force_input_mono

        self.teacher_model = teacher_model

        if optimizer_configs is None:
            optimizer_configs ={
                "autoencoder": {
                    "optimizer": {
                        "type": "AdamW",
                        "config": {
                            "lr": lr,
                            "betas": (.8, .99)
                        }
                    }
                },
                "discriminator": {
                    "optimizer": {
                        "type": "AdamW",
                        "config": {
                            "lr": lr,
                            "betas": (.8, .99)
                        }
                    }
                }

            }

        self.optimizer_configs = optimizer_configs

        if loss_config is None:
            scales = [2048, 1024, 512, 256, 128, 64, 32]
            hop_sizes = []
            win_lengths = []
            overlap = 0.75
            for s in scales:
                hop_sizes.append(int(s * (1 - overlap)))
                win_lengths.append(s)

            loss_config = {
                "discriminator": {
                    "type": "encodec",
                    "config": {
                        "n_ffts": scales,
                        "hop_lengths": hop_sizes,
                        "win_lengths": win_lengths,
                        "filters": 32
                    },
                    "weights": {
                        "adversarial": 0.1,
                        "feature_matching": 5.0,
                    }
                },
                "spectral": {
                    "type": "mrstft",
                    "config": {
                        "fft_sizes": scales,
                        "hop_sizes": hop_sizes,
                        "win_lengths": win_lengths,
                        "perceptual_weighting": True
                    },
                    "weights": {
                        "mrstft": 1.0,
                    }
                },
                "time": {
                    "type": "l1",
                    "config": {},
                    "weights": {
                        "l1": 0.0,
                    }
                }
            }

        self.loss_config = loss_config

        # Spectral reconstruction loss

        stft_loss_args = loss_config['spectral']['config']

        if self.autoencoder.out_channels == 2:
            self.sdstft = SumAndDifferenceSTFTLoss(sample_rate=sample_rate, **stft_loss_args)
            self.lrstft = MultiResolutionSTFTLoss(sample_rate=sample_rate, **stft_loss_args)
        else:
            self.sdstft = MultiResolutionSTFTLoss(sample_rate=sample_rate, **stft_loss_args)

        # Discriminator

        self.discriminator = EncodecDiscriminator(in_channels=self.autoencoder.out_channels, **loss_config['discriminator']['config'])

        self.gen_loss_modules = []

        # Adversarial and feature matching losses
        self.gen_loss_modules += [
            ValueLoss(key='loss_adv', weight=self.loss_config['discriminator']['weights']['adversarial'], name='loss_adv'),
            ValueLoss(key='feature_matching_distance', weight=self.loss_config['discriminator']['weights']['feature_matching'], name='feature_matching'),
        ]

        if self.teacher_model is not None:
            # Distillation losses

            stft_loss_weight = self.loss_config['spectral']['weights']['mrstft'] * 0.25
            self.gen_loss_modules += [
                AuralossLoss(self.sdstft, 'reals', 'decoded', name='mrstft_loss', weight=stft_loss_weight), # Reconstruction loss
                AuralossLoss(self.sdstft, 'decoded', 'teacher_decoded', name='mrstft_loss_distill', weight=stft_loss_weight), # Distilled model's decoder is compatible with teacher's decoder
                AuralossLoss(self.sdstft, 'reals', 'own_latents_teacher_decoded', name='mrstft_loss_own_latents_teacher', weight=stft_loss_weight), # Distilled model's encoder is compatible with teacher's decoder
                AuralossLoss(self.sdstft, 'reals', 'teacher_latents_own_decoded', name='mrstft_loss_teacher_latents_own', weight=stft_loss_weight) # Teacher's encoder is compatible with distilled model's decoder
            ]

        else:

            # Reconstruction loss
            self.gen_loss_modules += [
                AuralossLoss(self.sdstft, 'reals', 'decoded', name='mrstft_loss', weight=self.loss_config['spectral']['weights']['mrstft']),
            ]

            if self.autoencoder.out_channels == 2:

                # Add left and right channel reconstruction losses in addition to the sum and difference
                self.gen_loss_modules += [
                    AuralossLoss(self.lrstft, 'reals_left', 'decoded_left', name='stft_loss_left', weight=self.loss_config['spectral']['weights']['mrstft']/2),
                    AuralossLoss(self.lrstft, 'reals_right', 'decoded_right', name='stft_loss_right', weight=self.loss_config['spectral']['weights']['mrstft']/2),
                ]

        if self.loss_config['time']['weights']['l1'] > 0.0:
            self.gen_loss_modules.append(L1Loss(key_a='reals', key_b='decoded', weight=self.loss_config['time']['weights']['l1'], name='l1_time_loss'))

        if self.autoencoder.bottleneck is not None:
            self.gen_loss_modules += create_loss_modules_from_bottleneck(self.autoencoder.bottleneck, self.loss_config)

        self.losses_gen = MultiLoss(self.gen_loss_modules)

        self.disc_loss_modules = [
            ValueLoss(key='loss_dis', weight=1.0, name='discriminator_loss'),
        ]

        self.losses_disc = MultiLoss(self.disc_loss_modules)

        # Set up EMA for model weights
        self.autoencoder_ema = None

        self.use_ema = use_ema

        if self.use_ema:
            self.autoencoder_ema = EMA(
                self.autoencoder,
                ema_model=ema_copy,
                beta=0.9999,
                power=3/4,
                update_every=1,
                update_after_step=1
            )

        self.latent_mask_ratio = latent_mask_ratio

    def configure_optimizers(self):

        opt_gen = create_optimizer_from_config(self.optimizer_configs['autoencoder']['optimizer'], self.autoencoder.parameters())
        opt_disc = create_optimizer_from_config(self.optimizer_configs['discriminator']['optimizer'], self.discriminator.parameters())

        if "scheduler" in self.optimizer_configs['autoencoder'] and "scheduler" in self.optimizer_configs['discriminator']:
            sched_gen = create_scheduler_from_config(self.optimizer_configs['autoencoder']['scheduler'], opt_gen)
            sched_disc = create_scheduler_from_config(self.optimizer_configs['discriminator']['scheduler'], opt_disc)
            return [opt_gen, opt_disc], [sched_gen, sched_disc]

        return [opt_gen, opt_disc]

    def training_step(self, batch, batch_idx):
        reals, encoder_input, _ = _unpack_batch(batch)

        if self.global_step >= self.warmup_steps:
            self.warmed_up = True

        loss_info = {}

        loss_info["reals"] = reals

        if self.force_input_mono and encoder_input.shape[1] > 1:
            encoder_input = encoder_input.mean(dim=1, keepdim=True)

        loss_info["encoder_input"] = encoder_input

        data_std = encoder_input.std()

        if self.warmed_up and self.encoder_freeze_on_warmup:
            with torch.no_grad():
                latents, encoder_info = self.autoencoder.encode(encoder_input, return_info=True)
        else:
            latents, encoder_info = self.autoencoder.encode(encoder_input, return_info=True)

        loss_info["latents"] = latents

        loss_info.update(encoder_info)

        # Encode with teacher model for distillation
        if self.teacher_model is not None:
            with torch.no_grad():
                teacher_latents = self.teacher_model.encode(encoder_input, return_info=False)
                loss_info['teacher_latents'] = teacher_latents

        # Optionally mask out some latents for noise resistance
        if self.latent_mask_ratio > 0.0:
            mask = torch.rand_like(latents) < self.latent_mask_ratio
            latents = torch.where(mask, torch.zeros_like(latents), latents)

        decoded = self.autoencoder.decode(latents)

        loss_info["decoded"] = decoded

        if self.autoencoder.out_channels == 2:
            loss_info["decoded_left"] = decoded[:, 0:1, :]
            loss_info["decoded_right"] = decoded[:, 1:2, :]
            loss_info["reals_left"] = reals[:, 0:1, :]
            loss_info["reals_right"] = reals[:, 1:2, :]

        # Distillation
        if self.teacher_model is not None:
            with torch.no_grad():
                teacher_decoded = self.teacher_model.decode(teacher_latents)
                own_latents_teacher_decoded = self.teacher_model.decode(latents) #Distilled model's latents decoded by teacher
                teacher_latents_own_decoded = self.autoencoder.decode(teacher_latents) #Teacher's latents decoded by distilled model

                loss_info['teacher_decoded'] = teacher_decoded
                loss_info['own_latents_teacher_decoded'] = own_latents_teacher_decoded
                loss_info['teacher_latents_own_decoded'] = teacher_latents_own_decoded


        if self.warmed_up:
            loss_dis, loss_adv, feature_matching_distance = self.discriminator.loss(reals, decoded)
        else:
            loss_dis = torch.tensor(0.).to(reals)
            loss_adv = torch.tensor(0.).to(reals)
            feature_matching_distance = torch.tensor(0.).to(reals)

        loss_info["loss_dis"] = loss_dis
        loss_info["loss_adv"] = loss_adv
        loss_info["feature_matching_distance"] = feature_matching_distance

        opt_gen, opt_disc = self.optimizers()

        lr_schedulers = self.lr_schedulers()

        sched_gen = None
        sched_disc = None

        if lr_schedulers is not None:
            sched_gen, sched_disc = lr_schedulers

        # Train the discriminator
        if self.global_step % 2 and self.warmed_up:
            loss, losses = self.losses_disc(loss_info)

            log_dict = {
                'train/disc_lr': opt_disc.param_groups[0]['lr']
            }

            opt_disc.zero_grad()
            self.manual_backward(loss)
            opt_disc.step()

            if sched_disc is not None:
                # sched step every step
                sched_disc.step()

        # Train the generator
        else:

            loss, losses = self.losses_gen(loss_info)

            if self.use_ema:
                self.autoencoder_ema.update()

            opt_gen.zero_grad()
            self.manual_backward(loss)
            opt_gen.step()

            if sched_gen is not None:
                # scheduler step every step
                sched_gen.step()

            log_dict = {
                'train/loss': loss.detach(),
                'train/latent_std': latents.std().detach(),
                'train/data_std': data_std.detach(),
                'train/gen_lr': opt_gen.param_groups[0]['lr']
            }

        for loss_name, loss_value in losses.items():
            log_dict[f'train/{loss_name}'] = loss_value.detach()

        self.log_dict(log_dict, prog_bar=True, on_step=True)

        return loss

    def export_model(self, path, use_safetensors=False):
        if self.autoencoder_ema is not None:
            model = self.autoencoder_ema.ema_model
        else:
            model = self.autoencoder

        if use_safetensors:
            save_model(model, path)
        else:
            torch.save({"state_dict": model.state_dict()}, path)


class AutoencoderDemoCallback(pl.Callback):
    def __init__(
        self,
        demo_dl,
        demo_every=2000,
        sample_size=65536,
        sample_rate=48000,
        experiment_id: str = None,
        loss_config: dict = None,
    ):
        super().__init__()
        self.demo_every = demo_every
        self.demo_samples = sample_size
        self.demo_dl = iter(demo_dl)
        self.sample_rate = sample_rate
        self.last_demo_step = -1
        self.experiment_id = experiment_id
        self.loss_config = loss_config

    @rank_zero_only
    @torch.no_grad()
    def on_train_batch_end(self, trainer, module, outputs, batch, batch_idx):

        def _comp_average_dic(one_path_res, skip_zero_key=True):

            if not one_path_res:
                return {}

            keys = one_path_res[0].keys()
            sum_dic = {k: 0.0 for k in keys}
            valid_count = {k: 0 for k in keys}
            for res in one_path_res:
                if res is None or (skip_zero_key and res.get(skip_zero_key) == 0.0):
                    continue

                for k in keys:
                    if res[k] is not None and res[k] != 0:
                        sum_dic[k] += res[k]
                        valid_count[k] += 1

            average_dic = {k: sum_dic[k] / valid_count[k] if valid_count[k]>0 else 0 for k in keys}

            return average_dic

        def _get_all_score(audio_paths, num_demos, demo_samples, n_each_sample):

            from score_tools.vae_metrics import Scorer as VaeLossScorer
            predictor = VaeLossScorer(sample_rate=self.sample_rate, loss_config=self.loss_config)

            for audio_idx, audio_path in enumerate(audio_paths):
                input_infos = []
                for i in range(num_demos):
                    input_infos.append({
                        "path": audio_path,
                        "start_frame": (demo_samples * n_each_sample) * i,
                        "n_each_sample": n_each_sample,
                        "demo_samples": demo_samples,
                    })

                one_path_res_dic_list = []
                for input_info in input_infos:
                    res_dic = {
                        "path": input_info["path"],
                        "start_time": input_info["start_frame"] / self.sample_rate,
                        "end_time": (input_info["start_frame"] + input_info["demo_samples"]*input_info["n_each_sample"]) / self.sample_rate,
                        }
                    res_dic['vae_losses'] = predictor._compute_one_vae_losses(input_info)

                    one_path_res_dic_list.append(res_dic)

                avg_dic = {"path": audio_path, "Is_average": True}
                if one_path_res_dic_list is not None:
                    avg_dic["average VAE losses"] = {
                        "decoder_vs_lossless":{
                            "all_freq": _comp_average_dic([x['vae_losses']["decoder_vs_lossless"]["all_freq"] for x in one_path_res_dic_list]),
                            "high_freq": _comp_average_dic([x['vae_losses']["decoder_vs_lossless"]["high_freq"] for x in one_path_res_dic_list])
                        }
                    }

                    if n_each_sample == 3:
                        avg_dic["average VAE losses"]["mp3_compress_vs_lossless"] = {
                            "all_freq": _comp_average_dic([x['vae_losses']["mp3_compress_vs_lossless"]["all_freq"] for x in one_path_res_dic_list]),
                            "high_freq": _comp_average_dic([x['vae_losses']["mp3_compress_vs_lossless"]["high_freq"] for x in one_path_res_dic_list])
                        }

                    all_freq_res = avg_dic["average VAE losses"]["decoder_vs_lossless"]["all_freq"]
                    high_freq_res = avg_dic["average VAE losses"]["decoder_vs_lossless"]["high_freq"]
                    logger.info(
                          "average VAE losses of %s --- "
                          "decoder_vs_lossless[all_freq]-stft:%.3f, mel_stft: %.3f, mcd:%.3f, pesq: %.3f, stoi: %.3f   "
                          "decoder_vs_lossless[high_freq]-stft:%.3f, mel_stft: %.3f",
                          audio_path,
                          all_freq_res['stft'], all_freq_res['mel_stft'], all_freq_res['mcd'], all_freq_res['pesq'], all_freq_res['stoi'],
                          high_freq_res['stft'], high_freq_res['mel_stft'],
                          )

                one_path_res_dic_list.append(avg_dic)

                json_path = f"{audio_path.rsplit('.', 1)[0]}.json"
                with open(json_path, 'w', encoding='utf-8') as f:
                    json.dump(one_path_res_dic_list, f, ensure_ascii=False, indent=2)

        if (trainer.global_step - 1) % self.demo_every != 0 or self.last_demo_step == trainer.global_step:
            return

        self.last_demo_step = trainer.global_step

        module.eval()

        try:
            demo_reals, encoder_input, is_superres = _unpack_batch(next(self.demo_dl))

            encoder_input = encoder_input.to(module.device)

            if module.force_input_mono:
                encoder_input = encoder_input.mean(dim=1, keepdim=True)

            with torch.no_grad():
                model = module.autoencoder_ema.ema_model if module.use_ema else module.autoencoder
                latents = model.encode(encoder_input)
                fakes = model.decode(latents)

            # Interleave: [input, reconstructed, lossless_ref] per sample for super-res,
            # or [input, reconstructed] per sample for standard mode
            if is_superres:
                tracks = [encoder_input, fakes, demo_reals.to(module.device)]
            else:
                tracks = [demo_reals.to(module.device), fakes]
            reals_fakes = rearrange(tracks, 'i b d n -> d (b i n)')

            log_dict = {}

            experiment_id = self.experiment_id if self.experiment_id is not None else trainer.logger.experiment.id
            save_dir = os.path.join(trainer.default_root_dir, trainer.logger.experiment.project, experiment_id, "demo")
            os.makedirs(save_dir, exist_ok=True)
            filename = os.path.join(save_dir, f'recon_{trainer.global_step:08}.wav')

            reals_fakes = reals_fakes.to(torch.float32).clamp(-1, 1).mul(32767).to(torch.int16).cpu()
            torchaudio.save(filename, reals_fakes, self.sample_rate)

            log_dict['recon'] = wandb.Audio(filename, sample_rate=self.sample_rate, caption='Reconstructed')
            log_dict['embeddings_3dpca'] = pca_point_cloud(latents)
            log_dict['embeddings_spec'] = wandb.Image(tokens_spectrogram_image(latents))
            log_dict['recon_melspec_left'] = wandb.Image(audio_spectrogram_image(reals_fakes))

            trainer.logger.experiment.log(log_dict)

            n_each_sample = 3 if is_superres else 2
            start_score_time = time()
            _get_all_score([filename], num_demos=demo_reals.shape[0], demo_samples=self.demo_samples, n_each_sample=n_each_sample)
            logger.info("Score evaluation took %.2fs", time() - start_score_time)

        except Exception as e:
            logger.error("%s: %s", type(e).__name__, e)
            raise
        finally:
            module.train()

def create_loss_modules_from_bottleneck(bottleneck, loss_config):
    losses = []

    if isinstance(bottleneck, VAEBottleneck):
        try:
            kl_weight = loss_config['bottleneck']['weights']['kl']
        except (KeyError, TypeError):
            kl_weight = 1e-6

        kl_loss = ValueLoss(key='kl', weight=kl_weight, name='kl_loss')
        losses.append(kl_loss)

    return losses
