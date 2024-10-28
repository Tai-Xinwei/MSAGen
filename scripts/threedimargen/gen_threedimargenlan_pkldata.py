# -*- coding: utf-8 -*-
import json
import pickle
import re
from dataclasses import asdict

import torch
from pymatgen.core import Structure
from pymatgen.io.cif import CifWriter
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers.generation.configuration_utils import GenerationConfig

from sfm.data.threedimargen_data.dataset import MODE, ThreeDimARGenDataset
from sfm.data.threedimargen_data.tokenizer import ThreeDimARGenTokenizer
from sfm.logging import logger
from sfm.models.threedimargen.threedimargen_config import (
    ThreeDimARGenConfig,
    ThreeDimARGenInferenceConfig,
)
from sfm.models.threedimargen.threedimargenlan import ThreeDimARGenLanModel
from sfm.utils import arg_utils
from sfm.utils.cli_utils import cli
from sfm.utils.move_to_device import move_to_device


def convert_to_dict(
    path="/msralaphilly2/ml-la/yinxia/wu2/backup/SFM_for_material.20240430/instruct_mat_7b_beam4_06282024.pkl",
    space_group=True,
):
    with open(path, "rb") as f:
        data = pickle.load(f)
    res = []
    for i in range(len(data)):
        if len(data[i][1]) == 0:
            continue
        seq = data[i][1][0]
        # extract sequence in seq before <sg*> tag
        elements = re.findall(r"([A-Z][a-z]*)", seq)
        sites = [{"element": ele} for ele in elements]

        if space_group:
            sg_no = re.search(r"<sg(\d+)>", seq)
            if sg_no is None:
                logger.error(f"no space group found in {seq}")
            else:
                sg_no = int(sg_no.group(1))
                res.append({"id": i, "sites": sites, "space_group": {"no": sg_no}})
        else:
            res.append({"id": i, "sites": sites})
    return res


def get_dataset(inference_config, saved_config, tokenizer, space_group=True):
    data = convert_to_dict(inference_config.input_file, space_group=space_group)
    saved_config.space_group = space_group
    dataset = ThreeDimARGenDataset.from_dict(
        tokenizer,
        data,
        saved_config,
        shuffle=False,
        mode=MODE.INFER,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=inference_config.infer_batch_size,
        shuffle=False,
        collate_fn=dataset.collate,
        drop_last=False,
    )
    return dataset, dataloader


def infer(
    model,
    dataloader,
    dataset,
    tokenizer,
    gen_config,
    inference_config,
    space_group=True,
):
    index = 0
    with torch.no_grad():
        for batch in tqdm(dataloader):
            batch = move_to_device(batch, "cuda")
            ret = model.net.generate(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                generation_config=gen_config,
            )
            sentences = ret.sequences.cpu().numpy()
            ret = tokenizer.decode_batch(sentences)
            for i in range(len(ret)):
                sent, lattice, atom_coordinates = ret[i]
                mat_id = dataset.data[index]["id"]
                species = [site["element"] for site in dataset.data[index]["sites"]]
                if len(atom_coordinates) > len(species):
                    atom_coordinates = atom_coordinates[: len(species)]
                if space_group:
                    space_group = dataset.data[index]["space_group"]["no"]
                    try:
                        structure = Structure(
                            lattice=lattice,
                            species=species,
                            coords=atom_coordinates,
                        )
                        cif = CifWriter(structure)
                        cif.write_file(
                            f"{inference_config.output_file}/{mat_id}_{space_group}.cif"
                        )
                    except Exception as e:
                        logger.error(f"{e}\n{species}\n{atom_coordinates}\n{lattice}")
                else:
                    try:
                        structure = Structure(
                            lattice=lattice,
                            species=species,
                            coords=atom_coordinates,
                        )
                        cif = CifWriter(structure)
                        cif.write_file(f"{inference_config.output_file}/{mat_id}_0.cif")
                    except Exception as e:
                        logger.error(f"{e}\n{species}\n{atom_coordinates}\n{lattice}")
                index += 1


@cli(ThreeDimARGenConfig, ThreeDimARGenInferenceConfig)
def main(args):
    config = arg_utils.from_args(args, ThreeDimARGenConfig)
    inference_config = arg_utils.from_args(args, ThreeDimARGenInferenceConfig)

    checkpoints_state = torch.load(config.loadcheck_path, map_location="cpu")
    saved_args = checkpoints_state["args"]
    saved_config = arg_utils.from_args(saved_args, ThreeDimARGenConfig)
    saved_config.tokenizer = config.tokenizer

    saved_config.update(asdict(inference_config))

    tokenizer = ThreeDimARGenTokenizer.from_file(config.dict_path, saved_config)

    logger.info(saved_config)

    model = ThreeDimARGenLanModel(saved_config)
    model.eval()
    # model.half()

    logger.info(f"Loading model from {config.loadcheck_path}")

    model.load_pretrained_weights(config.loadcheck_path)
    model.cuda()

    gen_config = GenerationConfig(
        pad_token_id=tokenizer.padding_idx,
        eos_token_id=tokenizer.eos_idx,
        use_cache=True,
        max_length=saved_config.max_position_embeddings,
        return_dict_in_generate=True,
    )

    logger.info(f"infering from {inference_config.input_file}")

    # with space_group
    dataset, dataloader = get_dataset(
        inference_config, saved_config, tokenizer, space_group=True
    )
    logger.info(f"loaded {len(dataset)} samples from {inference_config.input_file}")
    infer(
        model,
        dataloader,
        dataset,
        tokenizer,
        gen_config,
        inference_config,
        space_group=True,
    )
    # without space_group
    dataset, dataloader = get_dataset(
        inference_config, saved_config, tokenizer, space_group=False
    )
    logger.info(f"loaded {len(dataset)} samples from {inference_config.input_file}")
    infer(
        model,
        dataloader,
        dataset,
        tokenizer,
        gen_config,
        inference_config,
        space_group=False,
    )


if __name__ == "__main__":
    main()
