'''
Copyright 2019 The Microsoft DeepSpeed Team
'''

import torch
import torch.nn as nn

import os
from deepspeed import comm as dist
from packaging import version as pkg_version
from collections import OrderedDict

from deepspeed.runtime import ZeROOptimizer
from deepspeed.runtime.fp16.loss_scaler import CreateLossScaler
from deepspeed.runtime.utils import (bwc_tensor_model_parallel_rank,
                                     get_global_norm,
                                     empty_cache,
                                     see_memory_usage,
                                     inf,
                                     is_model_parallel_parameter,
                                     align_dense_tensors,
                                     all_gather_dp_groups)

from deepspeed.runtime.zero.config import ZeroStageEnum
from deepspeed.runtime.zero.offload_config import OffloadDeviceEnum
from deepspeed.ops.adam import DeepSpeedCPUAdam
from deepspeed.utils import logger
from deepspeed.moe.utils import is_moe_param
from deepspeed.git_version_info import version
from deepspeed.runtime.zero.stage_1_and_2 import DeepSpeedZeroOptimizer

from deepspeed.runtime.constants import PIPE_REPLICATED
from deepspeed.accelerator import get_accelerator
from deepspeed.ops.op_builder import UtilsBuilder

from deepspeed.checkpoint.constants import (DS_VERSION,
                                            GROUP_PADDINGS,
                                            PARTITION_COUNT,
                                            SINGLE_PARTITION_OF_FP32_GROUPS,
                                            BASE_OPTIMIZER_STATE,
                                            CLIP_GRAD,
                                            ZERO_STAGE,
                                            PARAM_SLICE_MAPPINGS)
from deepspeed.utils import link_hp_params
from deepspeed.checkpoint import enable_universal_checkpoint

# Toggle this to true to enable correctness test
# with gradient partitioning and without
pg_correctness_test = False


def input(msg):
    return


def split_half_float_double(tensors):
    device_type = get_accelerator().device_name()
    dtypes = [
        "torch.{}.HalfTensor".format(device_type),
        "torch.{}.FloatTensor".format(device_type),
        "torch.{}.DoubleTensor".format(device_type),
        "torch.{}.BFloat16Tensor".format(device_type)
    ]
    buckets = []
    for i, dtype in enumerate(dtypes):
        bucket = [t for t in tensors if t.type() == dtype]
        if bucket:
            buckets.append(bucket)
    return buckets


def isclose(a, b, rtol=1e-09, atol=0.0):
    return abs(a - b) <= max(rtol * max(abs(a), abs(b)), atol)


def lcm(x, y):
    from fractions import gcd  # or can import gcd from `math` in Python 3
    return x * y // gcd(x, y)


def get_alignment_padding(tensor_list, alignment):
    num_elements = sum([tensor.numel() for tensor in tensor_list])
    remainder = num_elements % alignment
    return (alignment - remainder) if remainder else remainder


def move_to_cpu(tensor_list):
    for tensor in tensor_list:
        tensor.data = tensor.data.cpu()


def print_rank_msg(msg):
    print(f"rank {dist.get_rank()} - {msg}")


def _get_padded_tensor(src_tensor, size):
    if src_tensor.numel() >= size:
        return src_tensor
    padded_tensor = torch.zeros(size, dtype=src_tensor.dtype, device=src_tensor.device)
    slice_tensor = torch.narrow(padded_tensor, 0, 0, src_tensor.numel())
    slice_tensor.data.copy_(src_tensor.data)
    return padded_tensor


class CopilotZeroOptimizer(DeepSpeedZeroOptimizer):
    """
    DeepSpeedZeroOptimizer designed to reduce the memory footprint
    required for training large deep learning models.

    For more details please see ZeRO: Memory Optimization Towards Training A Trillion Parameter Models
    https://arxiv.org/abs/1910.02054

    For usage examples, refer to TODO: DeepSpeed Tutorial

    """
    def __init__(self,
                 init_optimizer,
                 param_names,
                 timers,
                 static_loss_scale=1.0,
                 dynamic_loss_scale=False,
                 dynamic_loss_args=None,
                 verbose=True,
                 contiguous_gradients=True,
                 reduce_bucket_size=500000000,
                 allgather_bucket_size=5000000000,
                 dp_process_group=None,
                 expert_parallel_group=None,
                 expert_data_parallel_group=None,
                 reduce_scatter=True,
                 overlap_comm=False,
                 cpu_offload=False,
                 mpu=None,
                 clip_grad=0.0,
                 communication_data_type=torch.float16,
                 postscale_gradients=True,
                 gradient_predivide_factor=1.0,
                 gradient_accumulation_steps=1,
                 ignore_unused_parameters=True,
                 partition_grads=True,
                 round_robin_gradients=False,
                 has_moe_layers=False,
                 fp16_master_weights_and_gradients=False,
                 elastic_checkpoint=False):

        if dist.get_rank() == 0:
            logger.info(f"Reduce bucket size {reduce_bucket_size}")
            logger.info(f"Allgather bucket size {allgather_bucket_size}")
            logger.info(f"CPU Offload: {cpu_offload}")
            logger.info(f'Round robin gradient partitioning: {round_robin_gradients}')
        # The fused optimizer does all the work. We need this layer for two reason:
        # 1. maintain same user API from apex.fp16_utils
        # 2. keep common stuff here in case we need to add ne552w fused optimizer later

        self.elastic_checkpoint = elastic_checkpoint
        self.param_names = param_names
        self.mpu = mpu
        # differences from apex.fp16_utils:
        # - assume all model params in fp16
        # - assume all params requires grad
        # - flat by groups, not keeping state. TODO: remove state explicitly?
        # - master grad and unflat master weight never exist. TODO: a way to save out unflat master?
        if not get_accelerator().is_available():
            raise SystemError("Cannot use fp16 without accelerator.")
        self.optimizer = init_optimizer

        # Load pre-built or JIT compile (un)flatten ops
        util_ops = UtilsBuilder().load()
        self.flatten = util_ops.flatten
        self.unflatten = util_ops.unflatten

        # ZeRO stage 1 (False) or 2 (True)
        self.partition_gradients = partition_grads
        self.zero_stage_string = "ZeRO-2" if partition_grads else "ZeRO-1"

        self.timers = timers

        self.reduce_scatter = reduce_scatter

        self.overlap_comm = overlap_comm

        self.cpu_offload = cpu_offload

        self.deepspeed_adam_offload = cpu_offload

        self.device = get_accelerator().current_device_name() if not self.cpu_offload else 'cpu'

        self.dp_process_group = dp_process_group

        #expert parallel group
        self.ep_process_group = expert_parallel_group

        #data parallel group for experts
        self.expert_dp_process_group = expert_data_parallel_group

        #data parallel size for non-experts
        dp_size = dist.get_world_size(group=self.dp_process_group)

        #For MoE models this maybe different for different param group
        #It will be modified during MoE setup later in the init
        self.real_dp_process_group = [dp_process_group for i in range(len(self.optimizer.param_groups))]
        self.partition_count = [dp_size for i in range(len(self.optimizer.param_groups))]

        self.is_gradient_accumulation_boundary = True

        # CPU-Offload requires contiguous gradients
        self.contiguous_gradients = contiguous_gradients or cpu_offload

        self.has_moe_layers = has_moe_layers
        if self.has_moe_layers:
            self._configure_moe_settings()
        self._global_grad_norm = 0.

        if mpu is None:
            self.model_parallel_group = None
            self.model_parallel_world_size = 1
            self.model_parallel_rank = 0
        else:
            self.model_parallel_group = mpu.get_model_parallel_group()
            self.model_parallel_world_size = mpu.get_model_parallel_world_size()
            self.model_parallel_rank = bwc_tensor_model_parallel_rank(mpu)

        self.overflow = False
        self.clip_grad = clip_grad
        self.communication_data_type = communication_data_type
        self.gradient_predivide_factor = gradient_predivide_factor
        self.postscale_gradients = postscale_gradients
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.micro_step_id = 0
        self.ignore_unused_parameters = ignore_unused_parameters
        self.round_robin_gradients = round_robin_gradients

        self.extra_large_param_to_reduce = None
        self.fp16_master_weights_and_gradients = fp16_master_weights_and_gradients

        if self.fp16_master_weights_and_gradients:
            assert self.cpu_offload and type(self.optimizer) in [DeepSpeedCPUAdam], \
            f"fp16_master_and_gradients requires optimizer to support keeping fp16 master and gradients while keeping the optimizer states in fp32."\
            f"Currently only supported using ZeRO-Offload with DeepSpeedCPUAdam. But current setting is ZeRO-Offload:{self.cpu_offload} and optimizer type {type(self.optimizer)}." \
            f"Either disable fp16_master_weights_and_gradients or enable {self.zero_stage_string} Offload with DeepSpeedCPUAdam."

        if self.reduce_scatter:
            valid_reduce_scatter_dtypes = (torch.float16, torch.bfloat16, torch.float32)
            assert self.communication_data_type in valid_reduce_scatter_dtypes, f"{self.zero_stage_string} supports {valid_reduce_scatter_dtypes} communication_data_type with reduce scatter enabled. Got: '{self.communication_data_type}'"
            assert self.gradient_predivide_factor == 1.0, "gradient_predivide_factor != 1.0 is not yet supported with {self.zero_stage_string} with reduce scatter enabled"
            assert self.postscale_gradients, "pre-scale gradients is not yet supported with {self.zero_stage_string} with reduce scatter enabled"

        # param flattened by groups
        self.bit16_groups = []
        self.bit16_groups_flat = []

        # param partitioned by data parallel degree
        # this will contain a list of equal sized tensors
        # each of which will be updated by a different process
        self.parallel_partitioned_bit16_groups = []

        # a single 32-bit partition of the parallel partitioned parameters
        # that this process will update
        self.single_partition_of_fp32_groups = []

        # param partition info

        # These are the parameters in each group that will not be updated by this process directly
        self.params_not_in_partition = []

        # These are the parameters that will be updated by this process directly
        self.params_in_partition = []

        # Offset from the first parameter in the the self.params_in_partition
        # the parameter boundaries may not align with partition boundaries
        # so we need to keep track of the offset
        self.first_offset = []

        # number of elements per partition in each group
        self.partition_size = []

        # align nccl all-gather send buffers to 4-byte boundary
        self.nccl_start_alignment_factor = 2  # 4-byte alignment/sizeof(fp16) = 2

        assert (
            allgather_bucket_size % self.nccl_start_alignment_factor == 0
        ), f"allgather_bucket_size must be a multiple of nccl_start_alignment_factor, {self.nccl_start_alignment_factor} "

        self.all_reduce_print = False
        self.dtype = self.optimizer.param_groups[0]['params'][0].dtype

        self.round_robin_bit16_groups = []
        self.round_robin_bit16_indices = []

        # Use different parallel to do all_to_all_reduce related things
        # padding on each partition for alignment purposes
        self.groups_padding = []
        # loop to deal with groups
        for i, param_group in enumerate(self.optimizer.param_groups):
            partition_id = dist.get_rank(group=self.real_dp_process_group[i])

            # push this group to list before modify
            # TODO: Explore simplification that avoids the extra book-keeping by pushing the reordered group
            trainable_parameters = [param for param in param_group['params'] if param.requires_grad]
            self.bit16_groups.append(trainable_parameters)

            # not sure why apex was cloning the weights before flattening
            # removing cloning here

            see_memory_usage(f"Before moving param group {i} to CPU")
            # move all the parameters to cpu to free up GPU space for creating flat buffer
            move_to_cpu(self.bit16_groups[i])
            empty_cache()
            see_memory_usage(f"After moving param group {i} to CPU", force=False)

            # Reorder group parameters for load balancing of gradient partitioning during backward among ranks.
            # This ensures that gradients are reduced in a fashion such that ownership round robins among the ranks.
            # For example, rather than 3 gradients (g_n+2, g_n+1, g_n) that are reduced consecutively belonging
            # to the same rank, instead they will belong to 3 ranks (r_m+2, r_m+1, r_m).
            if self.round_robin_gradients:
                round_robin_tensors, round_robin_indices = self._round_robin_reorder(
                    self.bit16_groups[i], dist.get_world_size(group=self.real_dp_process_group[i]))
            else:
                round_robin_tensors = self.bit16_groups[i]
                round_robin_indices = list(range(len(self.bit16_groups[i])))

            self.round_robin_bit16_groups.append(round_robin_tensors)
            self.round_robin_bit16_indices.append(round_robin_indices)

            # create flat buffer in CPU and move to GPU
            self.bit16_groups_flat.append(
                self.flatten_dense_tensors_aligned(
                    self.round_robin_bit16_groups[i],
                    self.nccl_start_alignment_factor * dist.get_world_size(group=self.real_dp_process_group[i])).to(
                        get_accelerator().current_device_name()))
            see_memory_usage(f"After flattening and moving param group {i} to GPU", force=False)

            # Record padding required for alignment
            if partition_id == dist.get_world_size(group=self.real_dp_process_group[i]) - 1:
                padding = self.bit16_groups_flat[i].numel() - sum(
                    [t.numel() for t in self.round_robin_bit16_groups[i]])
            else:
                padding = 0
            self.groups_padding.append(padding)

            if dist.get_rank(group=self.real_dp_process_group[i]) == 0:
                see_memory_usage(f"After Flattening and after emptying param group {i} cache", force=False)

            # set model bit16 weight to slices of flattened buffer
            self._update_model_bit16_weights(i)

            # divide the flat weights into near equal partition equal to the data parallel degree
            # each process will compute on a different part of the partition
            data_parallel_partitions = self.get_data_parallel_partitions(self.bit16_groups_flat[i], i)
            self.parallel_partitioned_bit16_groups.append(data_parallel_partitions)

            # verify that data partition start locations are 4-byte aligned
            for partitioned_data in data_parallel_partitions:
                assert (partitioned_data.data_ptr() % (2 * self.nccl_start_alignment_factor) == 0)

            # A partition of the fp32 master weights that will be updated by this process.
            # Note that the params in single_partition_of_fp32_groups is cloned and detached
            # from the origin params of the model.
            if not fp16_master_weights_and_gradients:
                self.single_partition_of_fp32_groups.append(self.parallel_partitioned_bit16_groups[i][partition_id].to(
                    self.device).clone().float().detach())
            else:
                self.single_partition_of_fp32_groups.append(self.parallel_partitioned_bit16_groups[i][partition_id].to(
                    self.device).clone().half().detach())

            # Set local optimizer to have flat params of its own partition.
            # After this, the local optimizer will only contain its own partition of params.
            # In that case, the local optimizer only saves the states(momentum, variance, etc.) related to its partition's params(zero stage1).
            self.single_partition_of_fp32_groups[
                i].requires_grad = True  # keep this in case internal optimizer uses it
            param_group['params'] = [self.single_partition_of_fp32_groups[i]]

            partition_size = len(self.bit16_groups_flat[i]) / dist.get_world_size(group=self.real_dp_process_group[i])
            params_in_partition, params_not_in_partition, first_offset = self.get_partition_info(
                self.round_robin_bit16_groups[i], partition_size, partition_id)

            self.partition_size.append(partition_size)
            self.params_in_partition.append(params_in_partition)
            self.params_not_in_partition.append(params_not_in_partition)
            self.first_offset.append(first_offset)

        for rank in range(dist.get_world_size()):
            if dist.get_rank() == rank:
                print(
                    f"Rank: {rank} partition count {self.partition_count} and sizes{[(p.numel(), self.is_moe_param_group[i] if hasattr(self, 'is_moe_param_group') else False) for i,p in enumerate(self.single_partition_of_fp32_groups)]} "
                )
                dist.barrier()

        self.reduce_bucket_size = int(reduce_bucket_size)
        self.allgather_bucket_size = int(allgather_bucket_size)

        self.reduction_event = get_accelerator().Event(enable_timing=False, blocking=False)
        self.reduction_stream = get_accelerator().Stream()
        self.cpu_computation_stream = get_accelerator().Stream()
        self.copy_grad_stream = get_accelerator().Stream()
        self.callback_queued = False

        self.param_dict = {}

        # map between param_id and bool to specify if a param is in this partition
        self.is_param_in_current_partition = {}

        self.grads_in_ipg_bucket = []
        self.params_in_ipg_bucket = []
        self.elements_in_ipg_bucket = 0
        self.params_already_reduced = []
        self._release_ipg_buffers()
        self.previous_reduced_grads = None
        self.ipg_bucket_has_moe_params = False

        # simplified param id
        self.param_id = {}

        #interesting code: unique ids being assigned to individual parameters
        largest_param_numel = 0
        count = 0
        for i, params_group in enumerate(self.bit16_groups):
            for param in params_group:
                unique_id = id(param)
                self.param_id[unique_id] = count
                self.param_dict[count] = param
                self.params_already_reduced.append(False)
                if param.numel() > largest_param_numel:
                    largest_param_numel = param.numel()
                count = count + 1

        for param_group in self.params_in_partition:
            for param in param_group:
                self.is_param_in_current_partition[self.get_param_id(param)] = True

        for param_group in self.params_not_in_partition:
            for param in param_group:
                self.is_param_in_current_partition[self.get_param_id(param)] = False

        if self.cpu_offload:
            self.accumulated_grads_in_cpu = {}
            self.norm_for_param_grads = {}
            self.local_overflow = False
            self.grad_position = {}
            self.temp_grad_buffer_for_cpu_offload = get_accelerator().pin_memory(
                torch.zeros(largest_param_numel, device=self.device, dtype=self.dtype))
            self.temp_grad_buffer_for_gpu_offload = torch.zeros(largest_param_numel,
                                                                device=get_accelerator().current_device_name(),
                                                                dtype=self.dtype)
            for i, params_group in enumerate(self.bit16_groups):
                self.get_grad_position(i, self.params_in_partition[i], self.first_offset[i], self.partition_size[i])

        # mapping from parameter to partition that it belongs to
        self.param_to_partition_ids = {}

        # stores if a partition has been reduced in this step
        self.is_partition_reduced = {}

        # number of grads in partition that still need to be computed
        self.remaining_grads_in_partition = {}

        # total number of grads in partition
        self.total_grads_in_partition = {}

        # stores if a grad in a partition has been computed or not
        self.is_grad_computed = {}

        # stores the offset at which a parameter gradient needs to be inserted in a partition
        self.grad_partition_insertion_offset = {}

        # the offset in the gradient at which it must be inserted at the beginning of the partition
        self.grad_start_offset = {}

        # will store the averaged gradients required by this partition
        self.averaged_gradients = {}

        # For cpu_offload, will store the averaged gradients required by this partition
        self.offload_gradient_dict = {}

        # store index of first parameter in each partition
        self.first_param_index_in_partition = {}

        # initializes all data structures for implementing gradient partitioning
        self.initialize_gradient_partitioning_data_structures()

        # resets the data structure value for the next backward propagation
        self.reset_partition_gradient_structures()

        # creates backward hooks for gradient partitioning
        if self.partition_gradients or self.overlap_comm:
            self.create_reduce_and_remove_grad_hooks()

        self.custom_loss_scaler = False
        self.external_loss_scale = None

        # we may have a way of fusing dynamic scale. Do not support for now
        self.loss_scaler = CreateLossScaler(dtype=self.dtype,
                                            static_loss_scale=static_loss_scale,
                                            dynamic_scaling=dynamic_loss_scale,
                                            dynamic_loss_args=dynamic_loss_args)
        self.dynamic_loss_scale = self.loss_scaler.dynamic

        see_memory_usage("Before initializing optimizer states", force=True)
        self.initialize_optimizer_states()
        see_memory_usage("After initializing optimizer states", force=True)

        if dist.get_rank() == 0:
            logger.info(f"optimizer state initialized")

        if dist.get_rank(group=self.dp_process_group) == 0:
            see_memory_usage(f"After initializing ZeRO optimizer", force=True)

        self._link_all_hp_params()
        self._enable_universal_checkpoint()

        self.freeze_param_dict = {}
        self.freeze_id2name = {}
        self._param_slice_mappings = self._create_param_mapping()


    def _create_param_mapping(self):
        param_mapping = []
        # count = 0
        for i, _ in enumerate(self.optimizer.param_groups):
            param_mapping_per_group = OrderedDict()
            for lp in self.bit16_groups[i]:
                if lp._hp_mapping is not None:
                    lp_name = self.param_names[lp]
                    if self._is_freeze(lp_name):
                        unique_id = id(lp)
                        count = self.param_id[unique_id]
                        self.freeze_param_dict[count] = 1
                        self.freeze_id2name[count] = lp_name
                    param_mapping_per_group[
                        lp_name] = lp._hp_mapping.get_hp_fragment_address()
                # count += 1
            param_mapping.append(param_mapping_per_group)
            
        return param_mapping


    def _is_freeze(self, para_name):
        #freeze lamma
        nl = para_name.split('.')[0]
        if int(nl) >= 40:
            return True
        else:
            return False

    def _customized_zero_grad(self, original_params, freeze_list=None):
        count = 0
        # for i, group in enumerate(original_param_groups):
        #     # self._update_model_bit16_weights(i)
        #     # print(self.bit16_groups[i])
        #     # print(len(self.bit16_groups[i]), len(self.params_in_partition[i])); exit()
        for param in original_params['params']:
            # if self.is_param_in_current_partition[self.get_param_id(param)]:
            
                # print("grad: ", param.grad); exit()
                # print("if has count:", count in self.freeze_param_dict)
            if self.get_param_id(param) in self.freeze_param_dict and param.grad is not None:
                param.grad.mul_(0.0)
                print("zeroing", self.freeze_id2name[count], param.grad)

            count += 1


    # def reduce_gradients(self, pipeline_parallel=False):
    #     world_size = dist.get_world_size(self.dp_process_group)
    #     my_rank = dist.get_rank(self.dp_process_group)

    #     # with PP we must create ipg buffer, since backward is handled outside zero
    #     if pipeline_parallel and self.contiguous_gradients:
    #         self.ipg_buffer = []
    #         buf_0 = torch.empty(int(self.reduce_bucket_size),
    #                             dtype=self.dtype,
    #                             device=get_accelerator().current_device_name())
    #         self.ipg_buffer.append(buf_0)
    #         self.ipg_index = 0

    #     # self._customized_zero_grad()
    #     if not self.overlap_comm:
    #         count = 0
    #         for i, group in enumerate(self.bit16_groups):
    #             for param in group:
    #                 if param.grad is not None:
    #                     self.reduce_ready_partitions_and_remove_grads(param, i)
                
    #                 count += 1
    #     # reduce any pending grads in either hook/non-hook case
    #     self.overlapping_partition_gradients_reduce_epilogue()


    def _optimizer_step(self, group_no):        
        original_param_groups = self.optimizer.param_groups
        self.optimizer.param_groups = [original_param_groups[group_no]]
        # Disabling this as the C++ side copy & synchronize is not working correctly
        #from deepspeed.ops.adam import DeepSpeedCPUAdam
        #if type(self.optimizer) == DeepSpeedCPUAdam and self.dtype == torch.half:
        #    self.optimizer.step(fp16_param_groups=[self.get_bit16_param_group(group_no)])
        #else:
        #    self.optimizer.step()
        self.optimizer.step()
        self.optimizer.param_groups = original_param_groups


    def step(self, closure=None):
        """
        Not supporting closure.
        """
        self.micro_step_id = -1

        see_memory_usage(f"In step before checking overflow")

        # First compute norm for all group so we know if there is overflow
        self.check_overflow()
        OPTIMIZER_ALLGATHER = 'optimizer_allgather'
        OPTIMIZER_GRADIENTS = 'optimizer_gradients'
        OPTIMIZER_STEP = 'optimizer_step'
        timer_names = [OPTIMIZER_ALLGATHER, OPTIMIZER_GRADIENTS, OPTIMIZER_STEP]

        prev_scale = self.loss_scale
        self._update_scale(self.overflow)
        if self.overflow:
            see_memory_usage('After overflow before clearing gradients')
            self.zero_grad(set_to_none=True)
            if self.cpu_offload:
                self.reset_cpu_buffers()
            else:
                self.averaged_gradients = {}

            see_memory_usage('After overflow after clearing gradients')

            self.start_timers(timer_names)
            self.stop_timers(timer_names)
            return

        # Step 1:- Calculate gradient norm using fp-16 grads
        if self.dtype == torch.float16:
            see_memory_usage('Before norm calculation')
            scaled_global_grad_norm = self.scaled_global_norm()
            self._global_grad_norm = scaled_global_grad_norm / prev_scale
            see_memory_usage('After norm before optimizer')

        # Step 2:- run optimizer and upscaling simultaneously
        for i, group in enumerate(self.bit16_groups):
            self.start_timers([OPTIMIZER_GRADIENTS])
            partition_id = dist.get_rank(group=self.real_dp_process_group[i])
            if self.cpu_offload:
                single_grad_partition = self.single_partition_of_fp32_groups[i].grad
                if self.dtype == torch.float16:
                    self.unscale_and_clip_grads([single_grad_partition], scaled_global_grad_norm)

                self.stop_timers([OPTIMIZER_GRADIENTS])
                self.start_timers([OPTIMIZER_STEP])
                self._optimizer_step(i)

                # Disabled, this is not currently working
                #from deepspeed.ops.adam import DeepSpeedCPUAdam
                #if not (type(self.optimizer) == DeepSpeedCPUAdam and self.dtype == torch.half):
                #    bit16_partitions = self.parallel_partitioned_bit16_groups[i]
                #    fp32_partition = self.single_partition_of_fp32_groups[i]
                #    bit16_partitions[partition_id].data.copy_(fp32_partition.data)
                bit16_partitions = self.parallel_partitioned_bit16_groups[i]
                fp32_partition = self.single_partition_of_fp32_groups[i]
                bit16_partitions[partition_id].data.copy_(fp32_partition.data)

                self.stop_timers([OPTIMIZER_STEP])
            else:
                # free gradients for all the parameters that are not updated by this process(ZeRO stage2)
                self.free_grad_in_param_list(self.params_not_in_partition[i])

                # create a flat gradients for parameters updated by this process
                # If we are last partition, ensure we have same size grads and partition size, if not pad with zero tensors
                if partition_id == dist.get_world_size(group=self.real_dp_process_group[i]) - 1:
                    single_grad_partition = self.flatten_dense_tensors_aligned(
                        self.averaged_gradients[i],
                        int(self.partition_size[i])).to(self.single_partition_of_fp32_groups[i].dtype)
                else:
                    single_grad_partition = self.flatten(self.averaged_gradients[i]).to(
                        self.single_partition_of_fp32_groups[i].dtype)
                assert single_grad_partition.numel() == self.partition_size[i], \
                    "averaged gradients have different number of elements that partition size {} {} {} {}".format(
                        single_grad_partition.numel(), self.partition_size[i], i, partition_id)

                self.single_partition_of_fp32_groups[i].grad = single_grad_partition
                # release all the gradient since we have already created a necessary copy in dp_grad_partition(ZeRO stage2)
                self.free_grad_in_param_list(self.params_in_partition[i])

                self.averaged_gradients[i] = None

                if self.dtype == torch.float16:
                    self.unscale_and_clip_grads([single_grad_partition], scaled_global_grad_norm)

                self.stop_timers([OPTIMIZER_GRADIENTS])

                # Step 3:- run the optimizer if no offloading
                self.start_timers([OPTIMIZER_STEP])
                self._optimizer_step(i)
                # Step 4:- get rid of the fp32 gradients. Not needed anymore
                self.single_partition_of_fp32_groups[i].grad = None
                del single_grad_partition
                bit16_partitions = self.parallel_partitioned_bit16_groups[i]
                fp32_partition = self.single_partition_of_fp32_groups[i]
                bit16_partitions[partition_id].data.copy_(fp32_partition.data)
                self.stop_timers([OPTIMIZER_STEP])

        see_memory_usage('After optimizer before all-gather')
        if self.cpu_offload:
            self.reset_cpu_buffers()

        self.start_timers([OPTIMIZER_ALLGATHER])
        # Gather the updated weights from everyone.
        # Then all partitions of the model parameters are updated and ready for next round forward.
        all_gather_dp_groups(partitioned_param_groups=self.parallel_partitioned_bit16_groups,
                             dp_process_group=self.real_dp_process_group,
                             start_alignment_factor=self.nccl_start_alignment_factor,
                             allgather_bucket_size=self.allgather_bucket_size)

        self.stop_timers([OPTIMIZER_ALLGATHER])

        # TODO: we probably don't need this? just to be safe
        for i in range(len(self.bit16_groups)):
            self._update_model_bit16_weights(i)

        self.log_timers(timer_names)
        see_memory_usage('After zero_optimizer step')

        return


