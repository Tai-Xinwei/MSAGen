# -*- coding: utf-8 -*-
import os
from dataclasses import replace

from sfm.data.dec_data.datasets import MixedTokenDataset
from sfm.models.decoder.deepfuse.config import (
    DataConfig,
    DecDeepFuseConfig,
    EntityDecoderType,
    TextDecoderType,
    llama2_7b_default_config,
    mix_gpt_default_config,
)
from sfm.models.decoder.deepfuse.model import DecDeepFuseModel
from sfm.pipeline.accelerator.trainer import Trainer
from sfm.utils import arg_utils
from sfm.utils.cli_utils import cli


@cli(DecDeepFuseConfig, DataConfig)
def main(args):
    config = arg_utils.from_args(args, DecDeepFuseConfig)

    assert (
        config.llama_model_type == TextDecoderType.LLaMA2_7B
    ), "let's train LLaMA2_7B first"
    assert (
        config.entity_decoder_model_type == EntityDecoderType.BioGPT
    ), "let's train BioGPT first"

    config = replace(config, **llama2_7b_default_config())
    config = replace(config, **mix_gpt_default_config())

    data_config = arg_utils.from_args(args, DataConfig)

    assert data_config.data_type == "text2mol", "let's train text2mol first"

    train_dataset = MixedTokenDataset.from_text_to_mol(
        mol_path=data_config.train_mol_path,
        text_path=data_config.train_txt_path,
        text_tokenizer=config.llama_model,
        entity_tokenizer=config.entity_decoder_model,
        max_text_len=config.max_text_len,
        max_entity_len=config.max_entity_len,
        show_example=True,
    )
    val_dataset = MixedTokenDataset.from_text_to_mol(
        mol_path=data_config.val_mol_path,
        text_path=data_config.val_text_path,
        text_tokenizer=config.llama_model,
        entity_tokenizer=config.entity_decoder_model,
        max_text_len=config.max_text_len,
        max_entity_len=config.max_entity_len,
    )

    num_gpus = int(os.environ.get("WORLD_SIZE", 1))
    config.iters_per_epoch = (
        len(train_dataset)
        // config.train_batch_size
        // config.gradient_accumulation_steps
        // num_gpus
    )

    model = DecDeepFuseModel(config)

    trainer = Trainer(config, model, train_dataset, val_dataset)
    trainer.train()


if __name__ == "__main__":
    main()
