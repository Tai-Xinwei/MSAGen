# -*- coding: utf-8 -*-
from collections import OrderedDict, namedtuple
from typing import Any, Dict, List, Mapping, Optional

import torch
from torch.nn.modules.module import _IncompatibleKeys

from megatron.model.module import MegatronModule
from sfm.logging import logger


class SFMModule(MegatronModule):
    """Megatron specific extensions of torch Module with support
    for pipelining."""

    def __init__(self, config=None, share_embeddings_and_output_weights=True):
        super(SFMModule, self).__init__(
            config=config,
            share_embeddings_and_output_weights=share_embeddings_and_output_weights,
        )
        self.config = config
        self.share_embeddings_and_output_weights = share_embeddings_and_output_weights

    def auto_partition_load_state_dict(
        self,
        state_dict: Mapping[str, Any],
        tp_model_size: int,
        tp_rank: int,
        strict: bool = True,
    ):
        """Load a partition of the model from a checkpoint."""

        if not isinstance(state_dict, Mapping):
            raise TypeError(
                "Expected state_dict to be dict-like, got {}.".format(type(state_dict))
            )

        missing_keys: List[str] = []
        unexpected_keys: List[str] = []
        error_msgs: List[str] = []

        getattr(state_dict, "_metadata", None)
        state_dict = OrderedDict(state_dict)

        for name, param in self.named_parameters():
            if name in state_dict:
                ckp_tensor = state_dict.pop(name, None)
                assert len(list(param.shape)) == len(
                    list(ckp_tensor.shape)
                ), f"model shape len {param.shape} != checkpoint shape len {ckp_tensor.shape}"

                # logger.info(
                #     f"name {name}, param.shape {param.shape}, ckp_tensor.shape {ckp_tensor.shape}"
                # )

                if (  # the bias is not sliced
                    len(list(param.shape)) == 1
                    and param.shape[0] == ckp_tensor.shape[0]
                ):
                    self._load(param, ckp_tensor, name, error_msgs)
                elif (  # the weight is not sliced
                    len(list(param.shape)) == 2
                    and param.shape[1] == ckp_tensor.shape[1]
                    and param.shape[0] == ckp_tensor.shape[0]
                ):
                    self._load(param, ckp_tensor, name, error_msgs)
                elif len(list(param.shape)) == 1 and (  # the weight is sliced in dim 0
                    param.shape[0] * tp_model_size == ckp_tensor.shape[0]
                    or param.shape[0] == ckp_tensor.shape[0] // tp_model_size + 1
                ):
                    partitioned_size = param.shape[0]
                    start_idx = tp_rank * partitioned_size
                    end_idx = (tp_rank + 1) * partitioned_size
                    if end_idx < ckp_tensor.shape[0]:
                        self._load(
                            param, ckp_tensor[start_idx:end_idx], name, error_msgs
                        )
                    else:
                        self._load(
                            param[: ckp_tensor.shape[0] - start_idx],
                            ckp_tensor[start_idx:],
                            name,
                            error_msgs,
                        )
                elif (  # the weight is sliced in dim 1
                    len(list(param.shape)) == 2
                    and param.shape[0] == ckp_tensor.shape[0]
                    and (
                        param.shape[1] * tp_model_size == ckp_tensor.shape[1]
                        or param.shape[1] == ckp_tensor.shape[1] // tp_model_size + 1
                    )
                ):
                    partitioned_size = param.shape[1]
                    start_idx = tp_rank * partitioned_size
                    end_idx = (tp_rank + 1) * partitioned_size
                    if end_idx < ckp_tensor.shape[1]:
                        self._load(
                            param, ckp_tensor[:, start_idx:end_idx], name, error_msgs
                        )
                    else:
                        self._load(
                            param[:, : ckp_tensor.shape[1] - start_idx],
                            ckp_tensor[:, start_idx:],
                            name,
                            error_msgs,
                        )
                elif (  # the weight is sliced in dim 0
                    len(list(param.shape)) == 2
                    and param.shape[1] == ckp_tensor.shape[1]
                    and (
                        param.shape[0] * tp_model_size == ckp_tensor.shape[0]
                        or param.shape[0] == ckp_tensor.shape[0] // tp_model_size + 1
                    )
                ):
                    partitioned_size = param.shape[0]
                    start_idx = tp_rank * partitioned_size
                    end_idx = (tp_rank + 1) * partitioned_size
                    if end_idx < ckp_tensor.shape[0]:
                        self._load(
                            param, ckp_tensor[start_idx:end_idx, :], name, error_msgs
                        )
                    else:
                        self._load(
                            param[: ckp_tensor.shape[0] - start_idx, :],
                            ckp_tensor[start_idx:, :],
                            name,
                            error_msgs,
                        )
                else:
                    raise Exception(
                        f"shape mismatch for {name}, param shape is {param.shape} and ckp_tensor shape is {ckp_tensor.shape}"
                    )
            else:
                missing_keys.append(name)

        unexpected_keys = list(state_dict.keys())

        if strict:
            if len(unexpected_keys) > 0:
                error_msgs.insert(
                    0,
                    "Unexpected key(s) in state_dict: {}. ".format(
                        ", ".join('"{}"'.format(k) for k in unexpected_keys)
                    ),
                )
            if len(missing_keys) > 0:
                error_msgs.insert(
                    0,
                    "Missing key(s) in state_dict: {}. ".format(
                        ", ".join('"{}"'.format(k) for k in missing_keys)
                    ),
                )

        if len(error_msgs) > 0:
            raise RuntimeError(
                "Error(s) in loading state_dict for {}:\n\t{}".format(
                    self.__class__.__name__, "\n\t".join(error_msgs)
                )
            )
        return _IncompatibleKeys(missing_keys, unexpected_keys)

    def _load(self, param, input_param, key, error_msgs):
        try:
            with torch.no_grad():
                param.copy_(input_param)
        except Exception as ex:
            error_msgs.append(
                'While copying the parameter named "{}", '
                "whose dimensions in the model are {} and "
                "whose dimensions in the checkpoint are {}, "
                "an exception occurred : {}.".format(
                    key, param.size(), input_param.size(), ex.args
                )
            )
