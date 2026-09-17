"""Factory functions for optimizers, schedulers, and training wrappers."""

import torch
from torch.nn import Parameter

from ..models.builder import create_model_from_config


class InverseLR(torch.optim.lr_scheduler._LRScheduler):
    """Inverse decay learning rate schedule with optional exponential warmup.

    inv_gamma is the number of steps required for the learning rate to decay to
    (1 / 2)**power of its original value.
    """

    def __init__(self, optimizer, inv_gamma=1., power=1., warmup=0., final_lr=0.,
                 last_epoch=-1, verbose=False):
        self.inv_gamma = inv_gamma
        self.power = power
        if not 0. <= warmup < 1:
            raise ValueError('Invalid value for warmup')
        self.warmup = warmup
        self.final_lr = final_lr
        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            import warnings
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.")
        return self._get_closed_form_lr()

    def _get_closed_form_lr(self):
        warmup = 1 - self.warmup ** (self.last_epoch + 1)
        lr_mult = (1 + self.last_epoch / self.inv_gamma) ** -self.power
        return [warmup * max(self.final_lr, base_lr * lr_mult)
                for base_lr in self.base_lrs]


def create_optimizer_from_config(optimizer_config: dict, parameters) -> torch.optim.Optimizer:
    """Instantiate an optimizer from a config dict with 'type' and 'config' keys."""
    optimizer_type = optimizer_config["type"]

    if optimizer_type == "FusedAdam":
        from deepspeed.ops.adam import FusedAdam
        optimizer = FusedAdam(parameters, **optimizer_config["config"])
    else:
        optimizer_fn = getattr(torch.optim, optimizer_type)
        optimizer = optimizer_fn(parameters, **optimizer_config["config"])
    return optimizer


def create_scheduler_from_config(scheduler_config: dict, optimizer):
    """Instantiate a learning rate scheduler from a config dict."""
    if scheduler_config["type"] == "InverseLR":
        scheduler_fn = InverseLR
    else:
        scheduler_fn = getattr(torch.optim.lr_scheduler, scheduler_config["type"])
    return scheduler_fn(optimizer, **scheduler_config["config"])


def create_training_wrapper_from_config(model_config: dict, training_config: dict, model):
    """Create a Lightning training wrapper (autoencoder or diffusion) from config."""
    model_type = model_config.get('model_type', None)
    assert model_type is not None, 'model_type must be specified in model config'
    assert training_config is not None, 'training_config must be provided'

    if model_type == 'autoencoder':
        from .autoencoders import AutoencoderTrainingWrapper

        ema_copy = None

        if training_config.get("use_ema", False):
            ema_copy = create_model_from_config(model_config)
            for name, param in model.state_dict().items():
                if isinstance(param, Parameter):
                    param = param.data
                ema_copy.state_dict()[name].copy_(param)

        teacher_model = training_config.get("teacher_model", None)
        if teacher_model is not None:
            teacher_model = create_model_from_config(teacher_model)
            teacher_model = teacher_model.eval().requires_grad_(False)

            teacher_model_ckpt = training_config.get("teacher_model_ckpt", None)
            if teacher_model_ckpt is not None:
                teacher_model.load_state_dict(torch.load(teacher_model_ckpt)["state_dict"])
            else:
                raise ValueError("teacher_model_ckpt must be specified if teacher_model is specified")

        return AutoencoderTrainingWrapper(
            model,
            lr=training_config["learning_rate"],
            warmup_steps=training_config.get("warmup_steps", 0),
            encoder_freeze_on_warmup=training_config.get("encoder_freeze_on_warmup", False),
            sample_rate=model_config["sample_rate"],
            loss_config=training_config.get("loss_configs", None),
            optimizer_configs=training_config.get("optimizer_configs", None),
            use_ema=training_config.get("use_ema", False),
            ema_copy=ema_copy,
            force_input_mono=training_config.get("force_input_mono", False),
            latent_mask_ratio=training_config.get("latent_mask_ratio", 0.0),
            teacher_model=teacher_model
        )
    elif model_type == 'diffusion_cond':
        from .diffusion import DiffusionCondTrainingWrapper
        return DiffusionCondTrainingWrapper(
            model,
            sample_rate=model_config["sample_rate"],
            lr=training_config.get("learning_rate", None),
            mask_padding=training_config.get("mask_padding", False),
            mask_padding_dropout=training_config.get("mask_padding_dropout", 0.0),
            use_ema=training_config.get("use_ema", True),
            log_loss_info=training_config.get("log_loss_info", False),
            optimizer_configs=training_config.get("optimizer_configs", None),
            cfg_dropout_prob=training_config.get("cfg_dropout_prob", 0.1),
            timestep_sampler=training_config.get("timestep_sampler", "uniform"),
            use_segalign=training_config.get("use_segalign", False),
            segalign_names=training_config.get("segalign_names", ["mert"]),
            segalign_depths=training_config.get("segalign_depths", None),
            segalign_coeff=training_config.get("segalign_coeff", 1.0),
        )
    else:
        raise NotImplementedError(f'Unknown model type: {model_type}')


def create_demo_callback_from_config(model_config: dict, training_config: dict, **kwargs):
    """Create a demo generation callback from config."""
    model_type = model_config.get('model_type', None)
    assert model_type is not None, 'model_type must be specified in model config'
    assert training_config is not None, 'training_config must be provided'

    demo_config = training_config.get("demo", {})
    dataset_config = kwargs.pop("dataset_config", {})
    experiment_id = kwargs.pop("experiment_id", None)

    if model_type == 'autoencoder':
        from .autoencoders import AutoencoderDemoCallback
        return AutoencoderDemoCallback(
            demo_every=demo_config.get("demo_every", 2000),
            sample_size=model_config["sample_size"],
            sample_rate=model_config["sample_rate"],
            experiment_id=experiment_id,
            loss_config=training_config.get("loss_configs", None),
            **kwargs
        )
    elif model_type == "diffusion_cond":
        from .diffusion import DiffusionCondDemoCallback

        return DiffusionCondDemoCallback(
            demo_every=demo_config.get("demo_every", 2000),
            sample_size=model_config["sample_size"],
            sample_rate=model_config["sample_rate"],
            demo_steps=demo_config.get("demo_steps", 250),
            num_demos=demo_config.get("num_demos", 0),
            demo_cfg_scales=demo_config.get("demo_cfg_scales", 1),
            demo_conditioning=demo_config.get("demo_cond", {}),
            demo_cond_from_batch=demo_config.get("demo_cond_from_batch", False),
            display_audio_cond=demo_config.get("display_audio_cond", False),
            demo_negative_cond=demo_config.get("demo_negative_cond", False),
            generate_demo=demo_config.get("generate_demo", True),
            valid_metrics=demo_config.get("valid_metrics", ["audiobox", "songeval", "repetition"]),
            experiment_id=experiment_id,
            dataset_func_args=dataset_config.get("custom_metadata_args", {})
        )
    else:
        raise NotImplementedError(f'Unknown model type: {model_type}')
