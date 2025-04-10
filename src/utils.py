"""
Modified by https://github.com/GAIR-NLP/MAYE/tree/master/maye/utils
For more details, visit: https://arxiv.org/abs/2504.02587
"""

import random
import json
import torch
import numpy as np

# Set random seed for reproducibility
def set_seed(seed):
    """Set random seed for all random number generators"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return seed

# File utilities
def open_jsonl(file_path, mode="r"):
    """Read data from a jsonl file"""
    with open(file_path, mode, encoding="utf-8") as fp:
        data = [json.loads(line) for line in fp.readlines()]
    return data

def save_jsonl(json_lines, file_path, mode="w", ensure_ascii=True):
    """Save data to a jsonl file"""
    with open(file_path, mode, encoding="utf-8") as fp:
        for json_line in json_lines:
            fp.write(json.dumps(json_line, ensure_ascii=ensure_ascii) + "\n")

# Padding utilities
def pad_sequence(list_of_tensors, dim, padding_value):
    """Pad a list of tensors to the same size along a dimension"""
    max_len = max(t.size(dim) for t in list_of_tensors)
    padded_list = []
    
    for t in list_of_tensors:
        pad_len = max_len - t.size(dim)
        t_padded = torch.nn.functional.pad(t, (0, pad_len), value=padding_value)
        padded_list.append(t_padded)
    
    padded_sequence = torch.cat(padded_list, dim=0)
    return padded_sequence

# Position ID generation for attention masks
def get_position_ids_from_padding_mask(padding_mask):
    """Generate position IDs from a padding mask"""
    return ((padding_mask.cumsum(-1) - 1) * padding_mask).to(torch.int)