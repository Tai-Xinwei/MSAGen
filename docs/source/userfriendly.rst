.. A4SFramework documentation master file, created by
   sphinx-quickstart on Mon Sep 25 05:01:57 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. _userfriendly:

User-friendly Interface
=======================

Cli decorator
-------------

.. automodule:: sfm.utils.cli_utils
   :members:
   :undoc-members:
   :show-inheritance:

The decorator :func:`sfm.utils.cli_utils.cli` is used to read args for the task. It is used as follows:

.. code-block:: python

   @cli(config1, config2, config3)
   def main(args):
      pass

   if __name__ == '__main__':
      main()

The decorator takes a list of config objects as arguments. The config objects are used to read the args. The examples to use config objects are shown in :mod:`sfm.tasks` (e.g., :mod:`sfm.tasks.graphormer.pretrain_graphormer`, :mod:`sfm.tasks.scigpt.pretrain_scigpt`).


Unified Trainer
---------------

.. autoclass:: sfm.pipeline.accelerator.trainer.Trainer
   :members:
   :undoc-members:
   :show-inheritance:

The trainer is used to train the model. It is used as follows:

.. code-block:: python

   @cli(config1, config2, config3)
   def main(args):

      # Define the dataset
      train_dataset = your_dataset(...)
      eval_dataset = your_dataset(...)

      # Define the model
      model = Model(args)

      trainer = Trainer(
         config=args,
         model=model,
         train_dataset=train_dataset,
         eval_dataset=eval_dataset,
      )

      trainer.train()

   if __name__ == '__main__':
      main()

The trainer takes args, model, and dataset as arguments. The model examples is defined in :mod:`sfm.models` (e.g., :mod:`sfm.models.graphormer.graphormer`, :mod:`sfm.models.scigpt.scigpt`). The dataset is defined in :mod:`sfm.data` (e.g., :mod:`sfm.data.mol_data`, :mod:`sfm.data.sci_data`).


Training Strategy
-----------------

.. autoclass:: sfm.pipeline.accelerator.dataclasses.TrainStrategy
   :members:
   :undoc-members:
   :show-inheritance:

The training strategy is used to define the training strategy. The trainer supports 9 modes: Single GPU, DDP (Distributed Data-Parallel), Zero0 (pure DeepSpeed), Zero1, Zero2, Zero3, ZeroInf (Zero-Infinity),
Pipeline parallelism and 3D parallelism. The training strategy can be used by the following code:

.. code-block:: bash

   torchrun --nproc_per_node=8 --use_env main.py --strategy Zero1



Required Module
---------------

.. autoclass:: sfm.pipeline.accelerator.model.Model
   :members:
   :undoc-members:
   :show-inheritance:

The required module is used to define the model. It is used as follows:

.. code-block:: python

   from sfm.pipeline.accelerator.model import Model
   class YourModel(Model):

      def __init__(self, args):
         super().__init__()
         self.args = args

         # Define the model
         self.model = your_model(...)

      def forward(self, batch):
         # Define the forward function
         return self.model(...)

      def compute_loss(self, pred, batch) -> ModelOutput:
         # Define the loss function
         pass

      def config_optimizer(
         self, model: Optional[nn.Module]
      ) -> Tuple[Optimizer, LRScheduler]:
         # Define the optimizer and lr_scheduler. If not defined, the default optimizer and lr_scheduler will be used.
         pass


The model takes args as arguments. The args is defined in :mod:`sfm.models` (e.g., :mod:`sfm.models.scigpt.scigpt`).
