.. A4SFramework documentation master file, created by
   sphinx-quickstart on Mon Sep 25 05:01:57 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. _trainargs:

Training Args
==================

This is a checklist to use Training Args in A4SFramework.


TrainStrategy
------------------

.. autoclass:: sfm.pipeline.accelerator.dataclasses.TrainStrategy
   :members:
   :undoc-members:
   :show-inheritance:

Seven different training strategies are supported in A4SFramework. You can choose one of them by setting the ``--strategy`` argument in scripts.


DistributedConfig
------------------

.. autoclass:: sfm.pipeline.accelerator.dataclasses.DistributedConfig
   :members:
   :undoc-members:
   :show-inheritance:


The DistributedConfig class, found in sfm.pipeline.accelerator.dataclasses, is designed to handle the configuration settings for distributed training.


TrainerConfig
------------------

.. autoclass:: sfm.pipeline.accelerator.dataclasses.TrainerConfig
   :members:
   :undoc-members:
   :show-inheritance:

The TrainerConfig class, found in sfm.pipeline.accelerator.dataclasses, serves as a comprehensive container for various configuration settings required during the training process.


TrainerState
------------------

.. autoclass:: sfm.pipeline.accelerator.dataclasses.TrainerState
   :members:
   :undoc-members:
   :show-inheritance:

The TrainerState class, found in sfm.pipeline.accelerator.dataclasses, is designed to keep track of the current state of the training process.
