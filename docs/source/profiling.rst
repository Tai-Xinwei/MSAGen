.. A4SFramework documentation master file, created by
   sphinx-quickstart on Mon Sep 25 05:01:57 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. _profiling:

Profiling Guide
==================

A4SFramework aims to identify the performance bottlenecks in the training process and provide guidance on how to optimize the training process.
This guide will provide a brief overview of the profiling tools and how to use them.


Implemented
-------------

Low-level profiling relies on the `torch.profiler` module. This module provides a way to profile the PyTorch model's performance. For the high-level profiling,
`Nvidia Nsight Systems Profiler <https://developer.nvidia.com/nsight-systems/>`__ is used. Below is a brief overview of the implemented functions and their usage.


sfm.pipeline.accelerator.trainer.profiler_init module
---------------------------------------
.. automodule:: sfm.pipeline.accelerator.trainer.profiler_init
   :members:
   :undoc-members:
   :show-inheritance:

The training loop in `sfm.pipeline.accelerator.trainer.train` function is wrapped with the profiler context manager.
`sfm.pipeline.accelerator.trainer.profiler_init` function is called before the training loop to initialize the profiler
with the scheduling options and the trace handler. Profiler's context manager accepts these parameters:

- `activities` - a list of activities to profile:
   - `ProfilerActivity.CPU` - PyTorch operators, TorchScript functions and
   user-defined code labels;
   - `ProfilerActivity.CUDA` - on-device CUDA kernels;
- `record_shapes` - whether to record shapes of the operator inputs;
- `profile_memory` - whether to report amount of memory consumed by model's Tensors;
- `with_stack` - whether to record source information (file and line number) for the ops;
- `with_flops` - whether to estimate the FLOPs;
- `with_modules` - whether to record module hierarchy (including function names);
corresponding to the callstack of the op. e.g. If module A's forward call's module B's forward which contains an aten::add op,
then aten::add's module hierarchy is A.B;

The `schedule` parameter takes a function as an argument to specify the profiling frequency. `on_trace_ready` parameter takes also
a function as an argument to specify the process at each time trace handler is called. Two trace handlers are included, first one
exports the profiling metrics to TensorBoard and the second one, being the custom function described below, exports the profiling
metrics to a CSV file viewable, i.e., via Chrome Browser.

sfm.pipeline.accelerator.trainer.custom_trace_handler module
---------------------------------------
.. automodule:: sfm.pipeline.accelerator.trainer.custom_trace_handler
   :members:
   :undoc-members:
   :show-inheritance:

One has the option to extend the custom trace handler to export the profiling metrics to any other format. At this stage,
this is outputted as a `.json` trace file. This trace handler is called at each time `torch.profiler.profile`'s `step()` method
is called, which happens at each batch update. However, for longer runs, the defined `schedule` returns an action for the profiler,
this way, trace handler is called at specific times during the training for the profiling. Below is an example of how to use the
schedule parameter to define the profiling frequency.

.. code::

   my_schedule = torch.profiler.schedule(
      wait=5,
      warmup=1,
      active=3,
      repeat=2,
      skip_first=10
   )

In the example above, profiler will skip the first 15 steps, spend the next step on the warm up,
actively record the next 3 steps, skip another 5 steps, spend the next step on the warm up, actively
record another 3 steps. Since the `repeat=2` parameter value is specified, the profiler will stop
the recording after the first two cycles (`repeat=0` would cause the recording to run for an indefinite time).

At the end of each cycle profiler calls the specified `on_trace_ready` function and passes itself as
an argument. This function is used to process the new trace - either by obtaining the table output or
by saving the output on disk as a trace file.

Usually, in large trainings, a complete epoch is too expensive to profile. Therefore, it is wise to limit the profiling to a few steps.


sfm.pipeline.accelerator.trainer.profiler_end module
---------------------------------------
.. automodule:: sfm.pipeline.accelerator.trainer.profiler_end
   :members:
   :undoc-members:
   :show-inheritance:

Final step prints and writes out the results in the `on_trace_ready` function to a simple `.dat` file in profiling folder,
specified by the user with `--prof_dir` argument (default folder is `./prof`).


Usage
-------------
Low-level profiling is embeeded into the Cli decorator :func:`sfm.utils.cli_utils.cli`. One can simply turn on the profiling by adding the
`--profiling` argument to the command line, as:

.. code-block:: bash

   python.py --profiling --ptensorboard --prof_dir ./prof_other


The `--prof_dir` argument is optional and specifies another folder where the profiling results are stored (default folder is `./prof`).
One has the option to disable the TensorBoard export by removing the `--ptensorboard` argument from the command line. For indivitual tracking of
functions and modules, one can use the context manager `torch.profiler.record_function`. In its default configuration, the complete profiling of the
front- and backpropagation is summaries profiled under `accelerator.train_step` using this context manager.

High-level profiling is enabled by wrapping the `torchrun` command, as given below.

.. code-block:: bash

   sudo nsys profile -t cuda,mpi --gpu-metrics-device all -o nsight_report --cuda-memory-usage=true --run-as <local_user> \
      torchrun <args>

This command will trace CUDA (and eventually MPI for inter-node communication if DeepSpeed is selected) activities on all GPUs with their CUDA memory usage,
and will save the profiling results in the `nsight_report` folder.
This command is executed with adminstrative privilidges to access the system-wide process tree, therefore, the `--run-as` argument is used to specify the local user
(check if this needed in your system via `nsys status --environment`). The `nsight_report` folder contains the profiling results in a `.nsys-rep` file, which can be
opened with the Nvidia Nsight Systems GUI. Here, each CUDA kernel, including NCCL kernels for intra-node communication between GPUs, alongside its memory usage
is visible for each GPU. A comprehensive guide to use the Cli is given in
`Nvidia Nsight Systems Profiler Cli Guide <https://docs.nvidia.com/nsight-systems/UserGuide/index.html#cli-profile-command-switch-options>`__.

Results
-------------

`PyTorch's official profiling guide <https://pytorch.org/tutorials/intermediate/tensorboard_profiler_tutorial.html#use-tensorboard-to-view-results-and-analyze-model-performance/>`__
is a good starting point, where this link also includes an extended information on viewing profiling results in TensorBoard.

It is worth mantioning that there would be as many seperate trace files, depending on the profiler's activity determined by the `active` argument.
Therefore, in a minimal profiling run, one can choose this `active` argument to be 1 (or equal to the number of gradient accumulation steps to see the inter and intra-node activity).

For a quick glance of the profiling results, one can view the `profiler_list.dat` file in the profiling folder.
This file contains the summary of the profiling results sorted by the CUDA time spent of each operator.

As mentioned above, the high-level profiling results are stored in the `.nsys-rep` file, which can be opened with the Nvidia Nsight Systems GUI.

If DeepSpeed is selected as the distributed backend (via `--strategy=Zero` argument of the torchrun command), extended DeepSpeed specific profiling results are stored
in the `profiler_ds.txt` file of the `prof_dir` folder.

If the memory usage is of interest, an overview of the memory is also written to the end of the `profiler_list.dat` file in the `prof_dir` folder. This memory usage is slightly
extended if, similarly, DeepSpeed is selected by using DeepSpeed's internal memory tracking implementation. Moreover, for a high-level memory debugging, DeepSpeed includes printing
out memory occupation status on-the-fly for front and backpropagation steps, which can be activated through `--debug` argument.
