# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging

import numpy as np
import torch.utils.data

try:
    from collections.abc import Iterable
except ImportError:
    from collections import Iterable

logger = logging.getLogger(__name__)


class EpochListening:
    """Mixin for receiving updates whenever the epoch increments."""

    @property
    def can_reuse_epoch_itr_across_epochs(self):
        """
        Whether we can reuse the :class:`fairseq.data.EpochBatchIterator` for
        this dataset across epochs.
        This needs to return ``False`` if the sample sizes can change across
        epochs, in which case we may need to regenerate batches at each epoch.
        If your dataset relies in ``set_epoch`` then you should consider setting
        this to ``False``.
        """
        return True

    def set_epoch(self, epoch):
        """Will receive the updated epoch number at the beginning of the epoch."""
        pass


class FairseqDataset(torch.utils.data.Dataset, EpochListening):
    """A dataset that provides helpers for batching."""

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def collater(self, samples):
        """Merge a list of samples to form a mini-batch.
        Args:
            samples (List[dict]): samples to collate
        Returns:
            dict: a mini-batch suitable for forwarding with a Model
        """
        raise NotImplementedError

    def num_tokens(self, index):
        """Return the number of tokens in a sample. This value is used to
        enforce ``--max-tokens`` during batching."""
        raise NotImplementedError

    def num_tokens_vec(self, indices):
        """Return the number of tokens for a set of positions defined by indices.
        This value is used to enforce ``--max-tokens`` during batching."""
        raise NotImplementedError

    def size(self, index):
        """Return an example's size as a float or tuple. This value is used when
        filtering a dataset with ``--max-positions``."""
        raise NotImplementedError

    def ordered_indices(self):
        """Return an ordered list of indices. Batches will be constructed based
        on this order."""
        return np.arange(len(self), dtype=np.int64)

    @property
    def supports_prefetch(self):
        """Whether this dataset supports prefetching."""
        return False

    def attr(self, attr: str, index: int):
        return getattr(self, attr, None)

    def prefetch(self, indices):
        """Prefetch the data required for this epoch."""
        raise NotImplementedError

    def get_batch_shapes(self):
        """
        Return a list of valid batch shapes, for example::
            [(8, 512), (16, 256), (32, 128)]
        The first dimension of each tuple is the batch size and can be ``None``
        to automatically infer the max batch size based on ``--max-tokens``.
        The second dimension of each tuple is the max supported length as given
        by :func:`fairseq.data.FairseqDataset.num_tokens`.
        This will be used by :func:`fairseq.data.FairseqDataset.batch_by_size`
        to restrict batch shapes. This is useful on TPUs to avoid too many
        dynamic shapes (and recompilations).
        """
        return None

    def collate_tokens(
        self,
        values,
        pad_idx,
        eos_idx=None,
        left_pad=False,
        move_eos_to_beginning=False,
        pad_to_length=None,
        pad_to_multiple=1,
        pad_to_bsz=None,
    ):
        """Convert a list of 1d tensors into a padded 2d tensor."""
        size = max(v.size(0) for v in values)
        size = size if pad_to_length is None else max(size, pad_to_length)
        if pad_to_multiple != 1 and size % pad_to_multiple != 0:
            size = int(((size - 0.1) // pad_to_multiple + 1) * pad_to_multiple)

        batch_size = len(values) if pad_to_bsz is None else max(len(values), pad_to_bsz)
        res = values[0].new(batch_size, size).fill_(pad_idx)

        def copy_tensor(src, dst):
            assert dst.numel() == src.numel()
            if move_eos_to_beginning:
                if eos_idx is None:
                    # if no eos_idx is specified, then use the last token in src
                    dst[0] = src[-1]
                else:
                    dst[0] = eos_idx
                dst[1:] = src[:-1]
            else:
                dst.copy_(src)

        for i, v in enumerate(values):
            copy_tensor(v, res[i][size - len(v) :] if left_pad else res[i][: len(v)])
        return res

    def batch_by_size(
        self,
        indices,
        max_tokens=None,
        max_sentences=None,
        required_batch_size_multiple=1,
    ):
        """
        Given an ordered set of indices, return batches according to
        *max_tokens*, *max_sentences* and *required_batch_size_multiple*.
        """
        # from fairseq.data import data_utils

        fixed_shapes = self.get_batch_shapes()
        if fixed_shapes is not None:

            def adjust_bsz(bsz, num_tokens):
                if bsz is None:
                    assert max_tokens is not None, "Must specify --max-tokens"
                    bsz = max_tokens // num_tokens
                if max_sentences is not None:
                    bsz = min(bsz, max_sentences)
                elif (
                    bsz >= required_batch_size_multiple
                    and bsz % required_batch_size_multiple != 0
                ):
                    bsz -= bsz % required_batch_size_multiple
                return bsz

            fixed_shapes = np.array(
                [
                    [adjust_bsz(bsz, num_tokens), num_tokens]
                    for (bsz, num_tokens) in fixed_shapes
                ]
            )

        try:
            num_tokens_vec = self.num_tokens_vec(indices).astype("int64")
        except NotImplementedError:
            num_tokens_vec = None

        return self.batch_by_size(
            indices,
            num_tokens_fn=self.num_tokens,
            num_tokens_vec=num_tokens_vec,
            max_tokens=max_tokens,
            max_sentences=max_sentences,
            required_batch_size_multiple=required_batch_size_multiple,
            fixed_shapes=fixed_shapes,
        )

    def _filter_by_size_dynamic(
        self, indices, size_fn, max_positions, raise_exception=False
    ):
        def compare_leq(a, b):
            return a <= b if not isinstance(a, tuple) else max(a) <= b

        def check_size(idx):
            if isinstance(max_positions, float) or isinstance(max_positions, int):
                return size_fn(idx) <= max_positions
            elif isinstance(max_positions, dict):
                idx_size = size_fn(idx)
                assert isinstance(idx_size, dict)
                intersect_keys = set(max_positions.keys()) & set(idx_size.keys())
                return all(
                    all(
                        a is None or b is None or a <= b
                        for a, b in zip(idx_size[key], max_positions[key])
                    )
                    for key in intersect_keys
                )
            else:
                # For MultiCorpusSampledDataset, will generalize it later
                if not isinstance(size_fn(idx), Iterable):
                    return all(size_fn(idx) <= b for b in max_positions)
                return all(
                    a is None or b is None or a <= b
                    for a, b in zip(size_fn(idx), max_positions)
                )

        def collect_filtered(function, iterable, filtered):
            """
            Similar to :func:`filter` but collects filtered elements in ``filtered``.
            Args:
                function (callable): function that returns ``False`` for elements that
                    should be filtered
                iterable (iterable): iterable to filter
                filtered (list): list to store filtered elements
            """
            for el in iterable:
                if function(el):
                    yield el
                else:
                    filtered.append(el)

        ignored = []
        itr = collect_filtered(check_size, indices, ignored)
        indices = np.fromiter(itr, dtype=np.int64, count=-1)
        return indices, ignored

    def filter_indices_by_size(self, indices, max_sizes):
        """
        Filter a list of sample indices. Remove those that are longer than
        specified in *max_sizes*.
        WARNING: don't update, override method in child classes
        Args:
            indices (np.array): original array of sample indices
            max_sizes (int or list[int] or tuple[int]): max sample size,
                can be defined separately for src and tgt (then list or tuple)
        Returns:
            np.array: filtered sample array
            list: list of removed indices
        """
        if isinstance(max_sizes, float) or isinstance(max_sizes, int):
            if hasattr(self, "sizes") and isinstance(self.sizes, np.ndarray):
                ignored = indices[self.sizes[indices] > max_sizes].tolist()
                indices = indices[self.sizes[indices] <= max_sizes]
            elif (
                hasattr(self, "sizes")
                and isinstance(self.sizes, list)
                and len(self.sizes) == 1
            ):
                ignored = indices[self.sizes[0][indices] > max_sizes].tolist()
                indices = indices[self.sizes[0][indices] <= max_sizes]
            else:
                indices, ignored = self._filter_by_size_dynamic(
                    indices, self.size, max_sizes
                )
        else:
            indices, ignored = self._filter_by_size_dynamic(
                indices, self.size, max_sizes
            )
        return indices, ignored

    @property
    def supports_fetch_outside_dataloader(self):
        """Whether this dataset supports fetching outside the workers of the dataloader."""
        return True


class FairseqIterableDataset(torch.utils.data.IterableDataset, EpochListening):
    """
    For datasets that need to be read sequentially, usually because the data is
    being streamed or otherwise can't be manipulated on a single machine.
    """

    def __iter__(self):
        raise NotImplementedError
