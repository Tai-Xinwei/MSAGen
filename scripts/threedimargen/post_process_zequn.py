# -*- coding: utf-8 -*-
import json
import os
import shutil

import ase
import ase.io
from tqdm import tqdm

SG_FILE = "/hai1/SFM/threedimargen/outputs/3dargenlan_v0.1_base_mp_nomad_qmdb_ddp_noniggli_layer24_head16_epoch50_warmup8000_lr1e-4_wd0.1_bs256/checkpoint_E40_mtgen_hit_correct_full.jsonl"
NOSG_FILE = "/hai1/SFM/threedimargen/outputs/3dargenlan_v0.1_base_mp_nomad_qmdb_ddp_noniggli_layer24_head16_epoch50_warmup8000_lr1e-4_wd0.1_bs256/checkpoint_E40_mtgen_hit_correct_full_nosg.jsonl"
INPUT_FILE = "/hai1/SFM/threedimargen/data/materials_data/mtgen_hit_correct_full.jsonl"
RAW_INPUT_FILE = (
    "/hai1/SFM/threedimargen/data/materials_data/mtgen_hit_correct_full.txt"
)
SAVE_DIR = "/hai1/SFM/threedimargen/outputs/mtgen_hit_correct_full"


def convert_to_ascii(fname):
    with open(fname, "r") as f:
        content = f.read()
    with open(fname, "w", encoding="ascii") as f:
        f.write(content)


def post_process():
    with open(RAW_INPUT_FILE, "r") as fr_raw:
        total_num = 0
        for line_raw in fr_raw:
            total_num += 1
    print(f"total_num: {total_num}")
    with open(SG_FILE, "r") as fr_sg, open(NOSG_FILE, "r") as fr_nosg, open(
        INPUT_FILE, "r"
    ) as fr_input:
        data_inputs = {}
        for line_input in fr_input:
            data_input = json.loads(line_input)
            data_inputs[data_input["id"]] = data_input
        data_outputs = {}
        for line_sg, line_nosg in zip(fr_sg, fr_nosg):
            data_sg = json.loads(line_sg)
            data_nosg = json.loads(line_nosg)
            assert data_sg["id"] == data_nosg["id"]
            data_outputs[data_sg["id"]] = (data_sg, data_nosg)

        last_id = 1
        for i in tqdm(range(1, total_num + 1)):
            if os.path.exists(
                os.path.join(SAVE_DIR, f"id{i}-A.cif")
            ) and os.path.exists(os.path.join(SAVE_DIR, f"id{i}-B.cif")):
                continue
            # outputs contains i
            if i in data_outputs:
                last_id = i
                data_sg, data_nosg = data_outputs[i]
            # outputs does not contain i, but in inputs, means i is filtered during infer due to length
            elif i in data_inputs:
                continue
            # outputs does not contain i, and not in inputs, means i is duplicated and can use last id
            else:
                try:
                    shutil.copy(
                        os.path.join(SAVE_DIR, f"id{last_id}-A.cif"),
                        os.path.join(SAVE_DIR, f"id{i}-A.cif"),
                    )
                    shutil.copy(
                        os.path.join(SAVE_DIR, f"id{last_id}-B.cif"),
                        os.path.join(SAVE_DIR, f"id{i}-B.cif"),
                    )
                except:
                    pass
                continue

            formula = [site["element"] for site in data_sg["sites"]]
            lattice_sg = data_sg["prediction"]["lattice"]
            atom_coordinates_sg = data_sg["prediction"]["coordinates"]
            lattice_nosg = data_nosg["prediction"]["lattice"]
            atom_coordinates_nosg = data_nosg["prediction"]["coordinates"]
            try:
                # Define a crystal structure using ASE's Atoms object
                structure_sg = ase.Atoms(
                    formula, scaled_positions=atom_coordinates_sg, cell=lattice_sg
                )
                structure_nosg = ase.Atoms(
                    formula, scaled_positions=atom_coordinates_nosg, cell=lattice_nosg
                )

                # Save the structure to a file in the .cif format, which can be read by VESTA
                ase.io.write(
                    os.path.join(SAVE_DIR, f"id{i}-A.cif"), structure_sg, format="cif"
                )
                ase.io.write(
                    os.path.join(SAVE_DIR, f"id{i}-B.cif"), structure_nosg, format="cif"
                )
                convert_to_ascii(os.path.join(SAVE_DIR, f"id{i}-A.cif"))
                convert_to_ascii(os.path.join(SAVE_DIR, f"id{i}-B.cif"))
            except Exception as e:
                print(f"Error: {i}: {e}")
                continue
    return


if __name__ == "__main__":
    post_process()
