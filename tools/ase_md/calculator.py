# -*- coding: utf-8 -*-
import json
import multiprocessing
import os
import warnings
from io import StringIO

import hydra
from omegaconf import DictConfig, OmegaConf, MISSING
from hydra import initialize, compose
from hydra.core.config_store import ConfigStore
import numpy as np
from datetime import timedelta

from ase import io
from ase.calculators.calculator import (
    Calculator,
    CalculatorSetupError,
    InputError,
    ReadError,
    all_changes,
)
from ase.config import cfg
from ase.units import Bohr, Hartree
from ase.atoms import Atoms

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List
from sfm.tasks.psm.ft_modules import PSM_FT_REGISTER, MDEnergyForceHead
from sfm.models.psm.loss.mae3ddiff import DiffMAE3dCriterions
from sfm.data.psm_data.dataset import SmallMolDataset
from sfm.models.psm.psm_config import PSMConfig
from sfm.models.psm.psmmodel import PSMModel
from sfm.pipeline.accelerator.dataclasses import DistributedTrainConfig, ModelOutput
from sfm.tasks.psm.finetune_psm_small_mol import SmallMolConfig
from sfm.data.psm_data.collator import collate_fn
from sfm.logging.loggers import get_logger
import torch

logger = get_logger()
class PSMCalculator(Calculator):

    implemented_properties = ['energy', 'forces']
    discard_results_on_any_change = True

    default_parameters = {}

    def __init__(self, restart=None, ignore_bad_restart=False,
                 label='psm-calc', atoms=None, command=None,
                 config_path="../config", config_name="config",
                 ref_energy: np.ndarray=None,
                 **kwargs):
        """
            Initializes the Calculator object.
            Parameters:
            - restart (str): Path to the restart file. Default is None.
            - ignore_bad_restart (bool): Whether to ignore bad restart files. Default is False.
            - label (str): Label for the calculation. Default is 'psm-calc'.
            - atoms (Atoms): Atoms object representing the system. Default is None.
            - command (str): Command to run the calculator. Default is None.
            - config_path (str): Path to the configuration file. Default is '../config'.
            - config_name (str): Name of the configuration. Default is 'config'.
            - ref_energy (np.ndarray): Reference energy for the calculation, unit must in ev. Default is None.
            - **kwargs: Additional keyword arguments.
            Returns:
            None
        """

        Calculator.__init__(self, restart=restart,
                            ignore_bad_restart=ignore_bad_restart, label=label,
                            atoms=atoms, command=command, **kwargs)

        self.config_path = config_path
        self.config_name = config_name
        self.ref_energy = ref_energy
        cs = ConfigStore.instance()
        cs.store(name=self.config_name, node=SmallMolConfig)
        self.config = self._load_config(config_path, config_name)
        torch.distributed.init_process_group(
            backend="NCCL",
            init_method="env://",
            world_size=self.config.world_size,
            rank=self.config.rank,
            timeout=timedelta(10)
        )
        torch.distributed.barrier()
        self._prepare_psm_model(self.config)


    def _load_config(self, config_path, config_name):
        with initialize(config_path=config_path):
            cfg = compose(config_name)
            cfg = OmegaConf.to_object(cfg)
        return cfg

    def _prepare_psm_model(self, config=None):
        if config is None:
            config = self.config

        finetune_module = None
        extra_collate_fn = None
        if config.psm_finetune_mode and config.finetune_module is not None:
            finetune_module = PSM_FT_REGISTER[config.finetune_module](config)
            extra_collate_fn = finetune_module.update_batched_data
        # Define model
        self.model = PSMModel(
            config, loss_fn=DiffMAE3dCriterions, psm_finetune_head=finetune_module
        )
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        logger.info(f"Init mode in device: {self.model.parameters().__next__().device}")

    def _create_pipeline(self, config):
        pass

    def _prepare_model_data(self, atoms: Atoms):
        coords = torch.from_numpy(atoms.get_positions().astype(np.float32)).reshape(-1, 3)
        num_atoms = coords.shape[0]
        out = {
            "sample_type": torch.tensor(0, dtype=torch.int32),
            "coords": coords,
            "forces": torch.zeros(num_atoms, 3, dtype=torch.float32),
            "num_atoms": torch.tensor(num_atoms, dtype=torch.int32),
            "token_type": torch.from_numpy(atoms.get_atomic_numbers().reshape(-1)),
            "idx": torch.tensor(0, dtype=torch.int32),
            "edge_index": torch.zeros(2, 1, dtype=torch.long),
            "energy": torch.tensor([0.0], dtype=torch.float32),
            "energy_per_atom": torch.zeros(1, dtype=torch.float32),
            "has_energy": torch.tensor([False], dtype=torch.bool),
            "has_forces": torch.tensor([False], dtype=torch.bool),
            "node_attr": torch.zeros(num_atoms, 118, dtype=torch.float32),
            "attn_bias": torch.zeros(num_atoms, num_atoms, dtype=torch.float32),
            "cell": torch.zeros((3, 3), dtype=torch.float32),
        }
        out = SmallMolDataset.generate_graph_feature(out, self.config.preprocess_2d_bond_features_with_cuda)
        out = collate_fn([out], multi_hop_max_dist=5)
        for k, v in out.items():
            if isinstance(v, torch.Tensor):
                out[k] = v.to(self.device)
        return out

    def _prepare_output(self, model_output: dict, atoms: Atoms)->dict:
        '''
        TODO: Add back the reference energy if used.
        '''
        energy = model_output['total_energy'].detach().cpu().numpy()[0]
        if self.ref_energy is not None:
            energy += self.ref_energy[atoms.get_atomic_numbers()].sum()
        results = {
            'energy': energy,
            'forces': model_output['forces'].detach().cpu().numpy()[0]
        }
        return results

    def read(self, label):
        """Read PSM outputs made from this ASE calculator
        """
        pass

    def calculate(self, atoms=None, properties=['energy'],
                  system_changes=all_changes, symmetry='c1'):

        for p in properties:
            if p not in self.implemented_properties:
                raise InputError('Property {} is not implemented'.format(p))

        Calculator.calculate(self, atoms=atoms)
        if self.atoms is None:
            raise CalculatorSetupError('An Atoms object must be provided to '
                                       'perform a calculation')
        atoms = self.atoms

        input_data = self._prepare_model_data(atoms)
        model_output = self.model(input_data)
        output = self._prepare_output(model_output, atoms)
        self.results = output
