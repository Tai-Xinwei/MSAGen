# -*- coding: utf-8 -*-

DEFAULT_DS_CONFIG = {
    "train_batch_size": 2,
    "gradient_accumulation_steps": 1,
    "zero_allow_untested_optimizer": True,
    "optimizer": {
        "type": "Adam",
        "params": {
            "lr": 0.00002,
            "betas": [0.9, 0.999],
            "eps": 1e-8,
            "weight_decay": 0.0,
        },
    },
    "scheduler": {
        "type": "WarmupDecayLR",
        "params": {
            "warmup_type": "linear",
            "total_num_steps": 1000000,
            "warmup_max_lr": 0.00002,
            "warmup_num_steps": 60000,
        },
    },
    "zero_optimization": {
        "stage": 1,
        "ignore_unused_parameters": True,
        "contiguous_gradients": True,
        "reduce_scatter": False,
    },
    "fp16": {"enabled": True, "auto_cast": False, "min_loss_scale": 0.0001},
    "gradient_clipping": 1.0,
    "steps_per_print": 100,
    "wandb": {
        "enabled": False,
        "team": "yourteam",
        "group": "yourgroup",
        "project": "yourproject",
    },
}
