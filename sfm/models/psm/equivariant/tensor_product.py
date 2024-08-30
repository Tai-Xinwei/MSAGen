# -*- coding: utf-8 -*-
from math import sqrt
from typing import List, Optional, Union

import torch
from e3nn import o3
from e3nn.o3._tensor_product._instruction import Instruction
from e3nn.util import prod
from e3nn.util.jit import compile_mode
from torch import nn


class Simple_TensorProduct(torch.nn.Module):
    def __init__(
        self,
        irreps_in1: o3.Irreps,
        irreps_in2: o3.Irreps,
        irreps_out: o3.Irreps,
        instructions: List[tuple],
        in1_var: Optional[Union[List[float], torch.Tensor]] = None,
        in2_var: Optional[Union[List[float], torch.Tensor]] = None,
        out_var: Optional[Union[List[float], torch.Tensor]] = None,
        irrep_normalization: str = "component",
        path_normalization: str = "element",
        internal_weights=True,
    ):
        super().__init__()

        self.irreps_in1 = o3.Irreps(irreps_in1)
        self.irreps_in2 = o3.Irreps(irreps_in2)
        self.irreps_out = o3.Irreps(irreps_out)

        instructions = [x if len(x) == 6 else x + (1.0,) for x in instructions]
        instructions = [
            Instruction(
                i_in1=i_in1,
                i_in2=i_in2,
                i_out=i_out,
                connection_mode=connection_mode,
                has_weight=has_weight,
                path_weight=path_weight,
                path_shape=(self.irreps_in1[i_in1].mul, self.irreps_in2[i_in2].mul),
            )
            for i_in1, i_in2, i_out, connection_mode, has_weight, path_weight in instructions
        ]

        if in1_var is None:
            in1_var = [1.0 for _ in range(len(self.irreps_in1))]

        if in2_var is None:
            in2_var = [1.0 for _ in range(len(self.irreps_in2))]

        if out_var is None:
            out_var = [1.0 for _ in range(len(self.irreps_out))]

        def num_elements(ins):
            return {
                "uvw": (
                    self.irreps_in1[ins.i_in1].mul * self.irreps_in2[ins.i_in2].mul
                ),
                "uvu": self.irreps_in2[ins.i_in2].mul,
            }[ins.connection_mode]

        normalization_coefficients = []
        for ins in instructions:
            mul_ir_in1 = self.irreps_in1[ins.i_in1]
            mul_ir_in2 = self.irreps_in2[ins.i_in2]
            mul_ir_out = self.irreps_out[ins.i_out]

            if irrep_normalization == "component":
                alpha = mul_ir_out.ir.dim
            if irrep_normalization == "norm":
                alpha = mul_ir_in1.ir.dim * mul_ir_in2.ir.dim

            if path_normalization == "element":
                x = sum(
                    in1_var[i.i_in1] * in2_var[i.i_in2] * num_elements(i)
                    for i in instructions
                    if i.i_out == ins.i_out
                )

            alpha /= x
            alpha *= out_var[ins.i_out]
            alpha *= ins.path_weight
            normalization_coefficients += [sqrt(alpha)]

        self.instructions = [
            Instruction(
                ins.i_in1,
                ins.i_in2,
                ins.i_out,
                ins.connection_mode,
                ins.has_weight,
                alpha,
                ins.path_shape,
            )
            for ins, alpha in zip(instructions, normalization_coefficients)
        ]

        self._in1_dim = self.irreps_in1.dim
        self._in2_dim = self.irreps_in2.dim

        self.weight_numel = sum(
            prod(ins.path_shape) for ins in self.instructions if ins.has_weight
        )

        self.internal_weights = internal_weights
        self.shared_weights = internal_weights
        if internal_weights and self.weight_numel > 0:
            assert self.shared_weights, "Having internal weights impose shared weights"
            self.weight = torch.nn.Parameter(torch.randn(self.weight_numel))
        else:
            # For TorchScript, there always has to be some kind of defined .weight
            self.register_buffer("weight", torch.Tensor())

    @torch.jit.unused
    def _prep_weights_python(
        self, weight: Optional[Union[torch.Tensor, List[torch.Tensor]]]
    ) -> Optional[torch.Tensor]:
        if isinstance(weight, list):
            weight_shapes = [
                ins.path_shape for ins in self.instructions if ins.has_weight
            ]
            if not self.shared_weights:
                weight = [
                    w.reshape(-1, prod(shape))
                    for w, shape in zip(weight, weight_shapes)
                ]
            else:
                weight = [
                    w.reshape(prod(shape)) for w, shape in zip(weight, weight_shapes)
                ]
            return torch.cat(weight, dim=-1)
        else:
            return weight

    def _get_weights(self, weight: Optional[torch.Tensor]) -> torch.Tensor:
        if not torch.jit.is_scripting():
            # If we're not scripting, then we're in Python and `weight` could be a List[Tensor]
            # deal with that:
            weight = self._prep_weights_python(weight)
        if weight is None:
            if self.weight_numel > 0 and not self.internal_weights:
                raise RuntimeError(
                    "Weights must be provided when the TensorProduct does not have `internal_weights`"
                )
            return self.weight
        else:
            if self.shared_weights:
                assert weight.shape == (self.weight_numel,), "Invalid weight shape"
            else:
                assert weight.shape[-1] == self.weight_numel, "Invalid weight shape"
                assert (
                    weight.ndim > 1
                ), "When shared weights is false, weights must have batch dimension"
            return weight

    def weight_views(
        self, weight: Optional[torch.Tensor] = None, yield_instruction: bool = False
    ):
        r"""Iterator over weight views for each weighted instruction.

        Parameters
        ----------
        weight : `torch.Tensor`, optional
            like ``weight`` argument to ``forward()``

        yield_instruction : `bool`, default False
            Whether to also yield the corresponding instruction.

        Yields
        ------
        If ``yield_instruction`` is ``True``, yields ``(instruction_index, instruction, weight_view)``.
        Otherwise, yields ``weight_view``.
        """
        weight = self._get_weights(weight)
        batchshape = weight.shape[:-1]
        offset = 0
        for ins_i, ins in enumerate(self.instructions):
            if ins.has_weight:
                flatsize = prod(ins.path_shape)
                this_weight = weight.narrow(-1, offset, flatsize).view(
                    batchshape + ins.path_shape
                )
                offset += flatsize
                if yield_instruction:
                    yield ins_i, ins, this_weight
                else:
                    yield this_weight

    def _sum_tensors(
        self, xs: List[torch.Tensor], shape: torch.Size, like: torch.Tensor
    ):
        if len(xs) > 0:
            out = xs[0]
            for x in xs[1:]:
                out = out + x
            return out
        return like.new_zeros(shape)

    def _main_left_right(self, x1s, x2s, weights):
        # 初始化输出形状
        empty = torch.empty((), device="cpu")
        output_shape = torch.broadcast_tensors(
            empty.expand(x1s.shape[:-1]), empty.expand(x2s.shape[:-1])
        )[0].shape
        del empty

        # 广播输入张量
        x1s, x2s = x1s.broadcast_to(output_shape + (-1,)), x2s.broadcast_to(
            output_shape + (-1,)
        )
        output_shape = output_shape + (self.irreps_out.dim,)
        x1s = x1s.reshape(-1, self.irreps_in1.dim)
        x2s = x2s.reshape(-1, self.irreps_in2.dim)
        batch_numel = x1s.shape[0]

        # 提取和重塑权重
        if self.weight_numel > 0:
            weights = weights.reshape(-1, self.weight_numel)

        # 提取每个不可约表示的输入
        x1_list = [
            x1s[:, i].reshape(batch_numel, mul_ir.mul, mul_ir.ir.dim)
            for i, mul_ir in zip(self.irreps_in1.slices(), self.irreps_in1)
        ]
        x2_list = [
            x2s[:, i].reshape(batch_numel, mul_ir.mul, mul_ir.ir.dim)
            for i, mul_ir in zip(self.irreps_in2.slices(), self.irreps_in2)
        ]

        outputs = []
        flat_weight_index = 0

        # 仅处理 "uvu" 的情况
        for ins in self.instructions:
            mul_ir_in1 = self.irreps_in1[ins.i_in1]
            mul_ir_in2 = self.irreps_in2[ins.i_in2]
            mul_ir_out = self.irreps_out[ins.i_out]

            # 跳过维度为 0 的情况
            # if mul_ir_in1.dim == 0 or mul_ir_in2.dim == 0 or mul_ir_out.dim == 0:
            #     continue

            x1 = x1_list[ins.i_in1]
            x2 = x2_list[ins.i_in2]

            # 计算 x1 和 x2 的外积
            xx = torch.einsum("zui,zvj->zuvij", x1, x2)
            # 获取 Wigner 3j 符号
            w3j = o3.wigner_3j(mul_ir_in1.ir.l, mul_ir_in2.ir.l, mul_ir_out.ir.l).to(
                x1s.device
            )
            # 取相应的weights
            w = weights[
                :, flat_weight_index : flat_weight_index + prod(ins.path_shape)
            ].reshape((-1,) + tuple(ins.path_shape))
            flat_weight_index += prod(ins.path_shape)
            # 计算结果
            result = torch.einsum("zuv,ijk,zuvij->zuk", w, w3j, xx)

            result = ins.path_weight * result
            outputs.append(result.reshape(batch_numel, mul_ir_out.dim))

        # 汇总输出
        outputs = [
            self._sum_tensors(
                [
                    out
                    for ins, out in zip(self.instructions, outputs)
                    if ins.i_out == i_out
                ],
                shape=(batch_numel, mul_ir_out.dim),
                like=x1s,
            )
            for i_out, mul_ir_out in enumerate(self.irreps_out)
            if mul_ir_out.mul > 0
        ]
        outputs = torch.cat(outputs, dim=1) if len(outputs) > 1 else outputs[0]
        return outputs.reshape(output_shape)

    def forward(self, x, y, weight: Optional[torch.Tensor] = None):
        assert x.shape[-1] == self._in1_dim, "Incorrect last dimension for x"
        assert y.shape[-1] == self._in2_dim, "Incorrect last dimension for y"
        weight = self.weight
        return self._main_left_right(x, y, weight)

        # def forward(self, x1 : torch.Tensor, x2 : torch.Tensor, weight : torch.Tensor) -> torch.Tensor:

    #     w = weight
    #     empty = torch.empty((), device = 'cpu')
    #     getattr_1 = x1.shape
    #     getitem = getattr_1[slice(None, -1, None)];  getattr_1 = None
    #     expand = empty.expand(getitem);  getitem = None
    #     getattr_2 = x2.shape
    #     getitem_1 = getattr_2[slice(None, -1, None)];  getattr_2 = None
    #     expand_1 = empty.expand(getitem_1);  empty = getitem_1 = None
    #     broadcast_tensors = torch.functional.broadcast_tensors(expand, expand_1);  expand = expand_1 = None
    #     getitem_2 = broadcast_tensors[0];  broadcast_tensors = None
    #     getattr_3 = getitem_2.shape;  getitem_2 = None
    #     add = getattr_3 + (-1,)
    #     broadcast_to = x1.broadcast_to(add);  x1 = add = None
    #     add_1 = getattr_3 + (-1,)
    #     broadcast_to_1 = x2.broadcast_to(add_1);  x2 = add_1 = None
    #     add_2 = getattr_3 + (7168,);  getattr_3 = None
    #     reshape = broadcast_to.reshape(-1, 4096);  broadcast_to = None
    #     reshape_1 = broadcast_to_1.reshape(-1, 3);  broadcast_to_1 = None
    #     getattr_4 = reshape.shape
    #     getitem_3 = getattr_4[0];  getattr_4 = None
    #     reshape_2 = w.reshape(-1, 3072);  w = None
    #     getitem_4 = reshape[(slice(None, None, None), slice(0, 1024, None))]
    #     reshape_3 = getitem_4.reshape(getitem_3, 1024, 1);  getitem_4 = None
    #     getitem_5 = reshape[(slice(None, None, None), slice(1024, 4096, None))];  reshape = None
    #     reshape_4 = getitem_5.reshape(getitem_3, 1024, 3);  getitem_5 = None
    #     reshape_5 = reshape_1.reshape(getitem_3, 1, 3);  reshape_1 = None
    #     getitem_6 = reshape_2[(slice(None, None, None), slice(0, 1024, None))]
    #     reshape_6 = getitem_6.reshape((1024, 1));  getitem_6 = None
    #     einsum = torch.functional.einsum('edb,eca->ecdab', reshape_5, reshape_3)
    #     reshape_7 = reshape_3.reshape(getitem_3, 1024);  reshape_3 = None
    #     einsum_1 = torch.functional.einsum('ca,ab->cab', reshape_7, reshape_6);  reshape_7 = reshape_6 = None
    #     einsum_2 = torch.functional.einsum('dbc,dca->dba', einsum_1, reshape_5);  einsum_1 = None
    #     reshape_8 = einsum_2.reshape(getitem_3, 3072);  einsum_2 = None
    #     getitem_7 = reshape_2[(slice(None, None, None), slice(1024, 2048, None))]
    #     reshape_9 = getitem_7.reshape((1024, 1));  getitem_7 = None
    #     mul = reshape_5 * 0.5773502691896258
    #     einsum_3 = torch.functional.einsum('dca,dba->dcb', mul, reshape_4);  mul = None
    #     einsum_4 = torch.functional.einsum('cba,ab->ca', einsum_3, reshape_9);  einsum_3 = reshape_9 = None
    #     reshape_10 = einsum_4.reshape(getitem_3, 1024);  einsum_4 = None
    #     getitem_8 = reshape_2[(slice(None, None, None), slice(2048, 3072, None))];  reshape_2 = None
    #     reshape_11 = getitem_8.reshape((1024, 1));  getitem_8 = None
    #     _w3j_1_1_1 = o3.wigner_3j(1,1,1).to(reshape_11.device)
    #     einsum_5 = torch.functional.einsum('dba,bc->dbac', reshape_4, reshape_11);  reshape_4 = reshape_11 = None
    #     mul_5 = reshape_5 * 1.7320508075688772;  reshape_5 = None
    #     tensordot = torch.functional.tensordot(mul_5, _w3j_1_1_1, dims = ((2,), (1,)), out = None);  mul_5 = _w3j_1_1_1 = None
    #     einsum_6 = torch.functional.einsum('edab,ecad->ecb', tensordot, einsum_5);  tensordot = einsum_5 = None
    #     reshape_12 = einsum_6.reshape(getitem_3, 3072);  einsum_6 = getitem_3 = None
    #     cat = torch.cat([reshape_10, reshape_8, reshape_12], dim = 1);  reshape_10 = reshape_8 = reshape_12 = None
    #     reshape_13 = cat.reshape(add_2);  cat = add_2 = None
    #     return reshape_13


# @compile_mode("script")
# class FeedForwardVec2Scalar(torch.nn.Module):
#     """
#     Use two (FCTP + Gate)
#     """

#     def __init__(
#         self,
#         irreps_node_input,
#         irreps_node_output,
#         proj_drop=0.1,
#     ):
#         super().__init__()
#         self.irreps_node_input = (
#             o3.Irreps(irreps_node_input)
#             if isinstance(irreps_node_input, str)
#             else irreps_node_input
#         )

#         self.irreps_node_output = (
#             o3.Irreps(irreps_node_output)
#             if isinstance(irreps_node_output, str)
#             else irreps_node_output
#         )

#         self.scalar_dim = self.irreps_node_input[0][0]  # l=0 scalar_dim

#         self.ir2scalar = Irreps2Scalar(
#             self.irreps_node_input[1:],
#             self.scalar_dim,
#             bias=True,
#             act="smoothleakyrelu",
#         )

#         self.slinear_1 = IrrepsLinear(
#             self.irreps_node_input, self.irreps_node_input, bias=True, act=None
#         )

#         self.slinear_2 = IrrepsLinear(
#             self.irreps_node_input, self.irreps_node_output, bias=True, act=None
#         )

#         self.scalar_linear = nn.Linear(self.scalar_dim, self.scalar_dim * 2)
#         nn.Sigmoid()

#         if proj_drop != 0.0:
#             self.proj_drop = EquivariantDropout(
#                 self.irreps_node_output, drop_prob=proj_drop
#             )

#     def forward(self, node_input, **kwargs):
#         """
#         irreps_in = o3.Irreps("128x0e+32x1e")
#         func =  FeedForwardNetwork(
#                 irreps_in,
#                 irreps_in,
#                 proj_drop=0.1,
#             )
#         out = func(irreps_in.randn(10,20,-1))
#         """
#         node_output = self.slinear_1(node_input)
#         scalar = node_output[..., : self.scalar_dim]
#         vec = node_output[..., self.scalar_dim :]
#         scalar1, scalar2 = torch.split(
#             self.scalar_linear(scalar), self.scalar_dim, dim=-1
#         )
#         scalar = scalar1 + self.ir2scalar(vec) * scalar2  # vec 2 scalar
#         node_output = torch.cat([scalar, vec], dim=-1)

#         node_output = self.slinear_2(node_output)

#         return node_output
