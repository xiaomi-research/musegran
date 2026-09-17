"""MuseGran training entry point.

Usage:
    python train.py --config configs/training/default.yaml
    python train.py --config configs/training/default.yaml --batch_size 16 --num_gpus 2
"""

import argparse
import json
import os
import random
import re
import shutil

import pytorch_lightning as pl
import torch
import yaml

from musegran.data.dataset import create_dataloader_from_config
from musegran.models import create_model_from_config
from musegran.models.builder import load_ckpt_state_dict, remove_weight_norm_from_model, copy_state_dict
from musegran.training import create_training_wrapper_from_config, create_demo_callback_from_config


def _load_config(path):
    with open(path) as f:
        if path.endswith(('.yaml', '.yml')):
            return yaml.safe_load(f)
        return json.load(f)


def parse_args():
    parser = argparse.ArgumentParser(description="MuseGran Training")
    parser.add_argument("--config", default="configs/training/default.yaml",
                        help="Path to training config (YAML or JSON)")
    known, unknown = parser.parse_known_args()

    config = _load_config(known.config)

    # Override config values from command line: --key value
    i = 0
    while i < len(unknown):
        if unknown[i].startswith("--"):
            key = unknown[i].lstrip("-")
            val = unknown[i + 1] if i + 1 < len(unknown) and not unknown[i + 1].startswith("--") else ""
            if key in config:
                orig = config[key]
                if isinstance(orig, bool):
                    val = val.lower() not in ("0", "false", "")
                elif isinstance(orig, int):
                    val = int(val)
                elif isinstance(orig, float):
                    val = float(val)
            config[key] = val
            i += 2 if val != "" else 1
        else:
            i += 1

    config["config_file"] = known.config
    return argparse.Namespace(**config)


def _get_step_from_filename(filename):
    """Extract step number from checkpoint filename like 'full-epoch=1-step=5000.ckpt'."""
    m = re.search(r'step=(\d+)', filename)
    return int(m.group(1)) if m else 0


class ExceptionCallback(pl.Callback):
    def on_exception(self, trainer, module, err):
        print(f'{type(err).__name__}: {err}')


class ModelConfigEmbedderCallback(pl.Callback):
    def __init__(self, model_config):
        self.model_config = model_config

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        checkpoint["model_config"] = self.model_config


class WeightsExportCallback(pl.Callback):
    """Export trainable model weights after full checkpoint is saved, then clean old full ckpts."""

    def __init__(self, every_n_train_steps, dirpath, prefix="model_weights"):
        self.every_n_train_steps = every_n_train_steps
        self.dirpath = dirpath
        self.prefix = prefix
        self._pending_cleanup = False

    def _cleanup_old_full_ckpts(self):
        full_ckpts = [f for f in os.listdir(self.dirpath) if f.endswith('.ckpt') and f.startswith('full-')]
        if len(full_ckpts) > 1:
            full_ckpts.sort(key=_get_step_from_filename)
            for old in full_ckpts[:-1]:
                os.remove(os.path.join(self.dirpath, old))

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if not self.dirpath:
            return

        if self._pending_cleanup:
            self._cleanup_old_full_ckpts()
            self._pending_cleanup = False

        if (trainer.global_step > 0
            and trainer.global_step % self.every_n_train_steps == 0):
            filename = f"{self.prefix}-epoch={trainer.current_epoch}-step={trainer.global_step}.ckpt"
            path = os.path.join(self.dirpath, filename)
            pl_module.export_trainable_weights(path)
            self._pending_cleanup = True


def main():
    args = parse_args()

    os.environ.setdefault("WANDB__SERVICE_WAIT", "300")
    if not os.environ.get("WANDB_API_KEY"):
        os.environ.setdefault("WANDB_MODE", "offline")

    seed = args.seed
    if os.environ.get("SLURM_PROCID") is not None:
        seed += int(os.environ.get("SLURM_PROCID"))

    random.seed(seed)
    torch.manual_seed(seed)

    model_config = _load_config(args.model_config)

    dataset_config = getattr(args, 'dataset', None) or {}
    training_config = getattr(args, 'training', None) or {}

    if "training" in model_config:
        import warnings
        warnings.warn("'training' in model_config is deprecated; move to training config file", stacklevel=2)
        training_config = {**model_config.pop("training"), **training_config}

    train_dl = create_dataloader_from_config(
        dataset_config,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        sample_rate=model_config["sample_rate"],
        sample_size=model_config["sample_size"],
        model_type=model_config.get("model_type", "diffusion_cond"),
        training_config=training_config,
        audio_channels=model_config.get("audio_channels", 2),
    )

    # Inject training-time overrides into model config
    conditioning_overrides = getattr(args, 'conditioning', None)
    if conditioning_overrides:
        model_config.setdefault("model", {}).setdefault("conditioning", {}).update(conditioning_overrides)

    lora_overrides = getattr(args, 'lora', None)
    if lora_overrides:
        model_config.setdefault("model", {}).setdefault("diffusion", {}).setdefault("config", {})["lora"] = lora_overrides

    # SegAlign: inject use_segalign + segalign_dims into DiT config from training config
    SEGALIGN_DIM_MAP = {"glap": 1024, "mert": 1024, "muq": 1024, "condss": 1024}
    if training_config.get("use_segalign", False):
        segalign_names = training_config.get("segalign_names", [])
        segalign_dims = [SEGALIGN_DIM_MAP[n] for n in segalign_names]
        dit_config = model_config.setdefault("model", {}).setdefault("diffusion", {}).setdefault("config", {})
        dit_config["use_segalign"] = True
        dit_config["segalign_dims"] = segalign_dims
    else:
        dit_config = model_config.get("model", {}).get("diffusion", {}).get("config", {})
        dit_config.pop("use_segalign", None)
        dit_config.pop("segalign_dims", None)

    model = create_model_from_config(model_config)

    wandb_logger = pl.loggers.WandbLogger(project="musegran")

    # Checkpoint directory
    if args.save_dir:
        experiment_id = None
        if hasattr(args, 'experiment_id') and args.experiment_id:
            experiment_id = args.experiment_id
        elif isinstance(wandb_logger.experiment.id, str):
            experiment_id = wandb_logger.experiment.id

        if experiment_id is None:
            checkpoint_dir = None
        else:
            checkpoint_dir = os.path.join(args.save_dir, experiment_id, "checkpoints")
            os.makedirs(checkpoint_dir, exist_ok=True)
            shutil.copyfile(args.model_config, os.path.join(checkpoint_dir, os.path.basename(args.model_config)))
            shutil.copyfile(args.config_file, os.path.join(checkpoint_dir, os.path.basename(args.config_file)))
    else:
        checkpoint_dir = None

    # Auto-resume from latest checkpoint
    resume_path = None
    if hasattr(args, 'experiment_id') and args.experiment_id and checkpoint_dir:
        ckpt_files = [f for f in os.listdir(checkpoint_dir) if f.endswith('.ckpt') and f.startswith('full-')]
        if ckpt_files:
            resume_path = os.path.join(checkpoint_dir, max(ckpt_files, key=_get_step_from_filename))

    if resume_path is None and getattr(args, 'resume_from', ''):
        resume_path = args.resume_from

    # Load pretrained weights
    if resume_path is None and getattr(args, 'pretrained_ckpt', ''):
        pretrained_sd = load_ckpt_state_dict(args.pretrained_ckpt)
        copy_state_dict(model, pretrained_sd)
        if model.pretransform is not None and not any('pretransform' in k for k in pretrained_sd):
            if not getattr(args, 'vae_ckpt', ''):
                print("WARNING: pretrained ckpt has no pretransform weights and vae_ckpt is empty.")

    if getattr(args, 'remove_pretransform_weight_norm', '') == "pre_load":
        remove_weight_norm_from_model(model.pretransform)

    if getattr(args, 'vae_ckpt', ''):
        model.pretransform.load_state_dict(load_ckpt_state_dict(args.vae_ckpt))

    if getattr(args, 'remove_pretransform_weight_norm', '') == "post_load":
        remove_weight_norm_from_model(model.pretransform)

    # Apply LoRA after pretrained weights are loaded
    if hasattr(model.model, 'apply_lora'):
        model.model.apply_lora()

    training_wrapper = create_training_wrapper_from_config(model_config, training_config, model)

    wandb_logger.watch(training_wrapper)

    ckpt_callback = pl.callbacks.ModelCheckpoint(
        every_n_train_steps=args.checkpoint_every_n_steps,
        dirpath=checkpoint_dir, save_top_k=1,
        filename="full-{epoch}-{step}",
    )

    demo_callback = create_demo_callback_from_config(
        model_config, training_config, demo_dl=train_dl,
        experiment_id=experiment_id, dataset_config=dataset_config,
    )

    weights_export_callback = WeightsExportCallback(
        every_n_train_steps=args.checkpoint_every_n_steps,
        dirpath=checkpoint_dir,
        prefix=getattr(args, 'weights_export_name', "model_weights"),
    )

    args_dict = vars(args)
    args_dict.update({"model_config": model_config, "dataset_config": dataset_config})
    wandb_logger.experiment.config.update(args_dict)

    strategy_name = getattr(args, 'strategy', '')
    if strategy_name:
        if strategy_name == "deepspeed":
            from pytorch_lightning.strategies import DeepSpeedStrategy
            strategy = DeepSpeedStrategy(
                stage=2, contiguous_gradients=True, overlap_comm=True,
                reduce_scatter=True, reduce_bucket_size=5e8, allgather_bucket_size=5e8,
                load_full_weights=True,
            )
        else:
            strategy = strategy_name
    else:
        strategy = 'ddp_find_unused_parameters_true' if args.num_gpus > 1 else "auto"

    trainer = pl.Trainer(
        devices=args.num_gpus,
        accelerator="gpu",
        num_nodes=args.num_nodes,
        strategy=strategy,
        precision=args.precision,
        accumulate_grad_batches=getattr(args, 'accum_batches', 1),
        callbacks=[ckpt_callback, weights_export_callback, demo_callback,
                   ExceptionCallback(), ModelConfigEmbedderCallback(model_config)],
        logger=wandb_logger,
        log_every_n_steps=10,
        max_epochs=getattr(args, 'max_epochs', None) or 10000000,
        default_root_dir=args.save_dir,
        gradient_clip_val=getattr(args, 'gradient_clip_val', 0.0),
        reload_dataloaders_every_n_epochs=0,
    )

    trainer.fit(training_wrapper, train_dl, ckpt_path=resume_path)


if __name__ == '__main__':
    main()
