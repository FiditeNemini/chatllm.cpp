"""
Convert Jacobian Lens to GGMM format
"""
import argparse
from ast import Dict, Tuple
from collections import OrderedDict
import copy
import json
import struct
import time, os
import io
import pickle
import re
from pathlib import Path
from enum import Enum, IntEnum
from pathlib import Path
from typing import IO, Any, Iterable, List, Optional, Tuple
import numpy as np
import math, gc

import torch
from torch import nn
from torch import _weight_norm

GGML_QK8_0 = 32
GGML_QK4_0 = 32
GGML_QK4_1 = 32
GGML_QK_K  = 256

GGML_MEM_ALIGN = 16

class GGMLType(Enum):
    F32 = 0
    F16 = 1
    Q4_0 = 2
    Q4_1 = 3
    Q8_0 = 8
    Q4_K = 12

class ModelType(Enum):
    ModelTypeTagDUMMY               = (0xffffffff)

g_tensor_types: list = []
g_model_meta = {}

def pad_to_len(l: list, to_len: int, v = 0) -> list:
    assert len(l) <= to_len
    n = len(l)
    return l + [v] * (to_len - n)

def quantize_q8_0(tensor: torch.Tensor) -> torch.CharTensor:
    # equivalent to ggml_quantize_q8_0 in ggml.c
    assert tensor.shape[tensor.ndim - 1] % GGML_QK8_0 == 0
    tensor = tensor.contiguous().view(-1, GGML_QK8_0)
    scale = tensor.abs().max(dim=-1, keepdim=True).values / ((1 << 7) - 1)
    tensor = (tensor / scale).round().clamp(min=-128, max=127).char()
    # add scale into each block
    tensor = torch.cat((scale.half().view(torch.int8), tensor), dim=-1)
    return tensor

def quantize_q4_0(tensor: torch.Tensor) -> torch.CharTensor:
    # equivalent to ggml_quantize_q4_0 in ggml.c
    assert tensor.shape[tensor.ndim - 1] % GGML_QK4_0 == 0
    tensor = tensor.contiguous().view(-1, GGML_QK4_0)
    abs_max_indices = tensor.abs().max(dim=-1, keepdim=True).indices
    max_values = torch.take_along_dim(tensor, abs_max_indices, dim=-1)
    scale = max_values / -8
    tensor = (tensor / scale + 8).round().clamp(min=0, max=15).char()
    # compress two int4 weights into a int8
    tensor = tensor[:, :16] | (tensor[:, 16:] << 4)
    # add scale into each block
    tensor = torch.cat((scale.half().view(torch.int8), tensor), dim=-1)
    return tensor

def quantize_q4_1(tensor: torch.Tensor) -> torch.CharTensor:
    # equivalent to ggml_quantize_q4_1 in ggml.c
    assert tensor.shape[tensor.ndim - 1] % GGML_QK4_1 == 0
    tensor = tensor.contiguous().view(-1, GGML_QK4_1)
    abs_max_indices = tensor.max(dim=-1, keepdim=True).indices
    max_values = torch.take_along_dim(tensor, abs_max_indices, dim=-1)
    abs_min_indices = tensor.min(dim=-1, keepdim=True).indices
    min_values = torch.take_along_dim(tensor, abs_min_indices, dim=-1)
    scale = (max_values - min_values) / 15
    tensor = ((tensor - min_values) / scale).round().clamp(min=0, max=15).char()
    # compress two int4 weights into a int8
    tensor = tensor[:, :16] | (tensor[:, 16:] << 4)
    # add scale into each block
    tensor = torch.cat((scale.half().view(torch.int8), min_values.half().view(torch.int8), tensor), dim=-1)
    return tensor

@torch.compile(fullgraph=True)
def batched_qkx2_quants(x: torch.Tensor, nmax: float, rmin: float, rdelta: float,
                        nstep: int, use_mad: bool):
    """
    x: (S, N) where N = 32
    Returns: scale (S,), offset (S,) = -min_x (positive), L (S, N)
    """
    S, N = x.shape
    # Weights: RMS + |x|
    av_x = torch.norm(x, dim=1) / math.sqrt(N)
    weights = av_x[:, None] + torch.abs(x)

    # Per‑row statistics
    min_data = torch.min(x, dim=1)[0]          # original min
    max_data = torch.max(x, dim=1)[0]          # original max
    sum_w = torch.sum(weights, dim=1)
    sum_x = torch.sum(weights * x, dim=1)

    # Clamp min_data > 0 to 0 (as in original)
    min_data = torch.where(min_data > 0, torch.zeros_like(min_data), min_data)

    # Handle degenerate rows (min == max)
    degenerate = (min_data == max_data)
    range_data = max_data - min_data
    range_safe = torch.where(degenerate, torch.ones_like(range_data), range_data)

    # Initial quantization using min_data as the centering offset
    iscale = nmax / range_safe
    scale = 1.0 / iscale
    L = (iscale[:, None] * (x - min_data[:, None])).round().clamp(0, nmax)
    offset = -min_data          # positive offset
    diff = scale[:, None] * L + min_data[:, None] - x
    diff = torch.abs(diff) if use_mad else torch.square(diff)
    best_mad = torch.sum(weights * diff, dim=1)

    if nstep > 0:
        inv_range = 1.0 / range_safe
        for step in range(nstep):
            # iscale uses constant min_data (never updated)
            iscale = (rmin + rdelta * step + nmax) * inv_range
            l = (iscale[:, None] * (x - min_data[:, None])).round().clamp(0, nmax)

            sum_l = torch.sum(weights * l, dim=1)
            sum_l2 = torch.sum(weights * l * l, dim=1)
            sum_xl = torch.sum(weights * l * x, dim=1)

            D = sum_w * sum_l2 - sum_l * sum_l
            D_inv = torch.where(D > 0, 1.0 / D, torch.zeros_like(D))

            this_scale = (sum_w * sum_xl - sum_x * sum_l) * D_inv
            this_min   = (sum_l2 * sum_x - sum_l * sum_xl) * D_inv   # this_min <= 0 normally

            # If this_min > 0, set to 0 and recompute scale
            pos_min_mask = this_min > 0
            sum_l2_safe = torch.where(sum_l2 == 0, torch.ones_like(sum_l2), sum_l2)
            new_scale_pos = sum_xl / sum_l2_safe
            this_scale = torch.where(pos_min_mask, new_scale_pos, this_scale)
            this_min   = torch.where(pos_min_mask, torch.zeros_like(this_min), this_min)

            diff_new = this_scale[:, None] * l + this_min[:, None] - x
            diff_new = torch.abs(diff_new) if use_mad else torch.square(diff_new)
            mad_new = torch.sum(weights * diff_new, dim=1)

            better = (D > 0) & (mad_new < best_mad)
            best_mad = torch.where(better, mad_new, best_mad)
            L = torch.where(better[:, None], l, L)
            scale = torch.where(better, this_scale, scale)
            offset = torch.where(better, -this_min, offset)   # offset = -min (positive)

    # Degenerate rows: set scale=0, L=0, offset=0
    scale = torch.where(degenerate, torch.zeros_like(scale), scale)
    L = torch.where(degenerate[:, None], torch.zeros(N, device=x.device, dtype=x.dtype), L)
    offset = torch.where(degenerate, torch.zeros_like(offset), offset)

    return scale, offset, L


@torch.compile(fullgraph=True)
def pack_q4k_scales(ls: torch.Tensor, lm: torch.Tensor) -> torch.Tensor:
    """
    ls, lm: (num_blocks, 8) uint8, each 0..63
    Returns: (num_blocks, 12) uint8 packed exactly as in GGML Q4_K.
    """
    B = ls.shape[0]
    out = torch.empty((B, 12), dtype=torch.uint8, device=ls.device)

    # j = 0..3: direct assignment
    out[:, 0] = ls[:, 0]
    out[:, 1] = ls[:, 1]
    out[:, 2] = ls[:, 2]
    out[:, 3] = ls[:, 3]
    out[:, 4] = lm[:, 0]
    out[:, 5] = lm[:, 1]
    out[:, 6] = lm[:, 2]
    out[:, 7] = lm[:, 3]

    # j = 4
    ls4 = ls[:, 4]; lm4 = lm[:, 4]
    out[:, 8] = (ls4 & 0xF) | ((lm4 & 0xF) << 4)
    out[:, 0] |= ((ls4 >> 4) << 6)
    out[:, 4] |= ((lm4 >> 4) << 6)

    # j = 5
    ls5 = ls[:, 5]; lm5 = lm[:, 5]
    out[:, 9] = (ls5 & 0xF) | ((lm5 & 0xF) << 4)
    out[:, 1] |= ((ls5 >> 4) << 6)
    out[:, 5] |= ((lm5 >> 4) << 6)

    # j = 6
    ls6 = ls[:, 6]; lm6 = lm[:, 6]
    out[:, 10] = (ls6 & 0xF) | ((lm6 & 0xF) << 4)
    out[:, 2] |= ((ls6 >> 4) << 6)
    out[:, 6] |= ((lm6 >> 4) << 6)

    # j = 7
    ls7 = ls[:, 7]; lm7 = lm[:, 7]
    out[:, 11] = (ls7 & 0xF) | ((lm7 & 0xF) << 4)
    out[:, 3] |= ((ls7 >> 4) << 6)
    out[:, 7] |= ((lm7 >> 4) << 6)

    return out

def quantize_q4_k(tensor: torch.Tensor, GGML_QK_K: int) -> torch.CharTensor:
    """
    Optimized Q4_K quantization, byte‑identical to the original.
    """
    orig_shape = tensor.shape
    assert orig_shape[-1] % GGML_QK_K == 0
    tensor = tensor.view(-1, GGML_QK_K)
    num_blocks = tensor.shape[0]
    subblocks_per_block = GGML_QK_K // 32   # = 8 for QK_K=256

    block_chunk_size: int = 8192 * 64

    # Pre‑allocate list to hold per‑block results (as uint8 tensors)
    block_results = []

    for start_block in range(0, num_blocks, block_chunk_size):
        end_block = min(start_block + block_chunk_size, num_blocks)
        chunk_blocks = end_block - start_block

        # Extract subblocks for this chunk: shape (chunk_blocks, 8, 32)
        chunk_subblocks = tensor[start_block:end_block].view(chunk_blocks, subblocks_per_block, 32)
        total_subblocks = chunk_blocks * subblocks_per_block
        subblock_data = chunk_subblocks.reshape(total_subblocks, 32)

        # Quantize all subblocks in this chunk
        scale_sub, offset_sub, L_sub = batched_qkx2_quants(
            subblock_data, nmax=15.0, rmin=-1.0, rdelta=0.1, nstep=20, use_mad=False
        )

        # Reshape back to block structure
        scale_sub = scale_sub.view(chunk_blocks, subblocks_per_block)
        offset_sub = offset_sub.view(chunk_blocks, subblocks_per_block)
        L_sub = L_sub.view(chunk_blocks, subblocks_per_block, 32)

        # Per‑block maxima
        max_scale = torch.max(scale_sub, dim=1)[0]
        max_offset = torch.max(offset_sub, dim=1)[0]

        # Integer scaling factors (0..63)
        inv_scale = torch.where(max_scale > 0, 63.0 / max_scale, torch.zeros_like(max_scale))
        inv_offset = torch.where(max_offset > 0, 64.0 / max_offset, torch.zeros_like(max_offset))

        ls = (inv_scale[:, None] * scale_sub).round().clamp(max=63).to(torch.uint8)
        lm = (inv_offset[:, None] * offset_sub).round().clamp(max=63).to(torch.uint8)

        # Pack scales
        scales_packed = pack_q4k_scales(ls, lm)   # (chunk_blocks, 12) uint8

        # d and dmin as half
        d = (max_scale / 63.0).half()
        dmin = (max_offset / 63.0).half()

        # Reconstruct per‑subblock scale and offset for final quantization
        rec_scale = (ls.float() * d[:, None]).view(chunk_blocks, subblocks_per_block, 1)
        rec_offset = (lm.float() * dmin[:, None]).view(chunk_blocks, subblocks_per_block, 1)

        # Final 4‑bit quantization
        L_quant = ((1.0 / rec_scale) * (chunk_subblocks + rec_offset)).round().clamp(0, 15).to(torch.uint8)

        # Pack nibbles
        L_quant = L_quant.view(chunk_blocks, subblocks_per_block // 2, 2, 32)
        L_packed = L_quant[:, :, 0, :] | (L_quant[:, :, 1, :] << 4)   # (chunk_blocks, 4, 32)
        L_packed = L_packed.view(chunk_blocks, -1)                     # (chunk_blocks, 128)

        # Convert d and dmin to bytes
        d_bytes = d.view(torch.uint8).view(chunk_blocks, 2)
        dmin_bytes = dmin.view(torch.uint8).view(chunk_blocks, 2)

        # Concatenate all parts for this chunk: 2+2+12+128 = 144 bytes per block
        chunk_result = torch.cat([d_bytes, dmin_bytes, scales_packed, L_packed], dim=1)
        block_results.append(chunk_result)

    # Combine all chunks and flatten to 1D int8
    if not block_results:
        return torch.empty(0, dtype=torch.int8, device=tensor.device)
    final = torch.cat(block_results, dim=0).view(torch.int8).flatten()

    return final

def dump_tensor(f, name: str, tensor: torch.Tensor, ggml_type: GGMLType):
    assert tensor.dtype == torch.float32

    # tensor name
    f.write(struct.pack("i", len(name.encode())))
    f.write(name.encode())

    # tensor shape & dtype
    f.write(struct.pack("i" * (2 + tensor.ndim), tensor.ndim, *tensor.shape, ggml_type.value))

    # tensor data
    try:
        if ggml_type == GGMLType.F32:
            tensor = tensor.float()
        elif ggml_type == GGMLType.F16:
            tensor = tensor.half()
        elif ggml_type == GGMLType.Q8_0:
            tensor = quantize_q8_0(tensor)
        elif ggml_type == GGMLType.Q4_0:
            tensor = quantize_q4_0(tensor)
        elif ggml_type == GGMLType.Q4_1:
            tensor = quantize_q4_1(tensor)
        elif ggml_type == GGMLType.Q4_K:
            tensor = quantize_q4_k(tensor, GGML_QK_K)
        else:
            raise NotImplementedError(f"Cannot dump tensor of dtype {tensor.dtype}")
    except Exception as e:
        raise Exception(f"Error dumping tensor {name} of shape {tensor.shape}: {e}")

    # align address
    aligned_pos = (f.tell() + (GGML_MEM_ALIGN - 1)) // GGML_MEM_ALIGN * GGML_MEM_ALIGN
    f.seek(aligned_pos)
    tensor.detach().numpy().tofile(f)

class AttributeDict(dict):
    def __getattr__(self, key):
        return self.__getitem__(key) if key in self else None

    __setattr__ = dict.__setitem__

def tabulate(data, headers = []) -> str:
    all = []
    all.append(headers)
    col_num = len(headers)
    for d in data:
        row = [str(x) for x in d]
        all.append(row)
        col_num = max(col_num, len(row))

    def get_col_width(n: int) -> int:
        nonlocal all
        r = 0
        for i in range(len(all)):
            row = all[i]
            if n < len(row): r = max(r, len(row[n]))
        return r

    widths = [get_col_width(i) for i in range(col_num)]
    sep = '+-' + '-+-'.join(['-' * w for w in widths]) + '-+\n'

    def make_row(i: int) -> int:
        nonlocal all, col_num, widths
        row = all[i]
        str_row = [str(row[n]) if n < len(row) else ' ' for n in range(col_num)]

        return '| ' + ' | '.join([f"{{:<{widths[n]}}}".format(str_row[n]) for n in range(col_num)]) + ' |\n'

    return sep + make_row(0) + sep + ''.join([make_row(i) for i in range(1, len(all))]) + sep

def tqdm(items, desc='') -> Iterable[any]:

    def print_progress_bar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 60, fill = '█', printEnd = "\r", auto_nl = True):
        percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
        filledLength = int(length * iteration // total)
        bar = fill * filledLength + '-' * (length - filledLength)
        print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
        if (iteration == total) and auto_nl:
            print()

    def format_time(t) -> str:
        return "{:.2f}s".format(t)

    total = len(items)
    t = time.perf_counter()
    for i, x in enumerate(items):
        yield x
        used = time.perf_counter() - t
        per_item = used / (i + 1)
        remain = (total - i - 1) * per_item
        print_progress_bar(i + 1, total, prefix=desc, suffix=f"({i}/{total}) {format_time(per_item)}/it rem: {format_time(remain)}")

def tensor_type_fallback(shape, ndim: int, ggml_type: GGMLType):
    match ggml_type:
        case GGMLType.Q8_0:
            if shape[ndim - 1] % GGML_QK8_0 == 0:
                return ggml_type
            else:
                return GGMLType.F16
        case GGMLType.Q4_0 | GGMLType.Q4_1:
            if shape[ndim - 1] % GGML_QK4_0 == 0:
                return ggml_type
            else:
                return GGMLType.F16
        case GGMLType.Q4_K:
            if shape[ndim - 1] % GGML_QK_K == 0:
                return ggml_type
            else:
                return tensor_type_fallback(shape, ndim, GGMLType.Q8_0)
        case _:
            return GGMLType.F16

def tensor_quantization_type(name: str, tensor: torch.Tensor, def_type: GGMLType) -> GGMLType:
    global g_tensor_types

    # 1d weight: convert it to float32
    if tensor.ndim <= 1:
        return GGMLType.F32

    assert tensor.ndim in {2, 3, 4}, f'unsupported: {name}.ndim = {tensor.ndim}'

    t = def_type
    for (pat, _t) in g_tensor_types:
        if re.match(pat, name):
            t = _t
            break

    t = tensor_type_fallback(tensor.shape, tensor.ndim, ggml_type=t)
    return t

def dump_state_dict(f, tensor_dict, ggml_type):
    tensor_info = []
    this_round = list(tensor_dict.keys())

    for name in tqdm(this_round, desc="Dumping ..."):
        tensor: torch.Tensor = tensor_dict[name]

        tensor = tensor.float()

        tensor_ggml_type = tensor_quantization_type(name, tensor, def_type=ggml_type)

        dump_tensor(f, name, tensor, tensor_ggml_type)
        tensor_info.append((name, tensor.shape, tensor_ggml_type.name))

    gc.collect()

    print(tabulate(tensor_info, headers=["name", "shape", "dtype"]))

class JLensConverter:
    FILE_VERSION = 1
    MODEL_TYPE   = ModelType.ModelTypeTagDUMMY

    @staticmethod
    def dump_config(f, config, ggml_type):
        config_values = [
            ggml_type.value,
            -1,
            config.hidden_size,
            -1,
            -1,
            -1,
            -1,
            -1,
            -1,
            -1,
            -1,
        ]
        f.write(struct.pack("i" * len(config_values), *config_values))

    @classmethod
    def convert(cls, config, tensor_dict, ggml_type, save_path):

        def write_size_to_pos(f, offset):
            size = f.tell()
            f.seek(offset)
            f.write(struct.pack("i", size))
            f.seek(0, 2)

        # convert all weights to fp16
        with open(save_path, "wb") as f:
            f.write(b"ggmm")  # magic
            GGMM_VER = 1
            f.write(struct.pack("i" * 4, GGMM_VER, 0, 0, 0))

            meta_bytes = json.dumps(g_model_meta, ensure_ascii=False).encode()
            rounded_len = ((len(meta_bytes) + 3) // 4) * 4
            meta_bytes += b'\x00' * (rounded_len - len(meta_bytes))
            f.write(meta_bytes)
            write_size_to_pos(f, 8)

            f.write(struct.pack("II", cls.MODEL_TYPE.value, cls.FILE_VERSION))  # model type & version
            cls.dump_config(f, config, ggml_type)
            write_size_to_pos(f, 12)
            write_size_to_pos(f, 16)

            dump_state_dict(f, tensor_dict, ggml_type)

        print(f"{cls.MODEL_TYPE.name} GGML model saved to {save_path}")


def main():
    global g_tensor_types

    parser = argparse.ArgumentParser("chatllm-convert")
    parser.add_argument("-i", "--model_name_or_path", type=str)
    parser.add_argument("-a", "--arch", type=str, default='')
    parser.add_argument("-l", "--lora_model_name_or_path", type=str, default=None)
    parser.add_argument("-o", "--save_path", type=Path)
    parser.add_argument("-t", "--type", type=str, default="q8_0", choices=["f32", "f16", "q8_0", "q4_0", "q4_1", "q4_k"])
    parser.add_argument("-tt", "--tensor-type", nargs=2, action='append', default=[], help='custom quantization for specific tensors. -tt pattern1 type1 -tt pattern2 type2')
    parser.add_argument("-n", "--name", type=str, required=True, help='model name in English')
    parser.add_argument("--native_name", type=str, default='', help='model native name')
    args = parser.parse_args()

    g_model_meta['model_name'] = args.name
    g_model_meta['model_native_name'] = args.native_name

    for l in reversed(args.tensor_type):
        g_tensor_types.append((l[0], GGMLType[l[1].upper()]))

    ggml_type = GGMLType[args.type.upper()]

    config = AttributeDict({})

    content = torch.load(args.model_name_or_path, map_location=torch.device('cpu'), weights_only=True)
    print(content.keys())
    tensor_dict = {}
    for i in content['source_layers']:
        tensor_dict[f"lens.{i}.weight"] = content['J'][i]

    config = {
        'hidden_size': content['d_model'],
        'source_layers': content['source_layers'],
    }

    g_model_meta['config.json'] = config

    JLensConverter.convert(AttributeDict(config), tensor_dict, ggml_type, args.save_path)

if __name__ == "__main__":
    main()
