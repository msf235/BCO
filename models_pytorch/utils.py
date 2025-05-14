import argparse
import os
import numpy as np
import torch
import torch.nn as nn

parser = argparse.ArgumentParser()
parser.add_argument(
    "--input_filename",
    default="input/demonstration.txt",
    help="the demonstration inputs",
)
parser.add_argument(
    "--mode", choices=["train", "test"], required=True, help="train or test mode"
)
parser.add_argument(
    "--model_dir", required=True, help="where to save/restore the model"
)

parser.add_argument(
    "--max_episodes", type=int, default=1000, help="the number of training episodes"
)
parser.add_argument(
    "--M", type=int, default=1000, help="the number of post demonstration examples"
)
parser.add_argument(
    "--batch_size", type=int, default=32, help="number of examples in a batch"
)
parser.add_argument(
    "--lr", type=float, default=0.002, help="initial learning rate for Adam"
)

parser.add_argument(
    "--save_freq",
    type=int,
    default=100,
    help="save model every save_freq iterations (0 to disable)",
)
parser.add_argument(
    "--print_freq",
    type=int,
    default=50,
    help="print loss/reward every print_freq iterations (0 to disable)",
)
parser.add_argument(
    "--continuous",
    action="store_true",
    help="if set, treat actions as continuous real-valued vectors",
)

args = parser.parse_args()


def weight_initializer(tensor: torch.Tensor, std: float = 0.1):
    """
    Truncated normal initializer with given standard deviation.
    """
    # PyTorch provides trunc_normal_; requires torch ≥1.10
    with torch.no_grad():
        nn.init.trunc_normal_(tensor, mean=0.0, std=std, a=-2 * std, b=2 * std)


def bias_initializer(tensor: torch.Tensor, val: float = 0.01):
    """
    Constant initializer for biases.
    """
    with torch.no_grad():
        nn.init.constant_(tensor, val)


def get_shuffle_idx(num: int, batch_size: int):
    """
    Split indices [0..num) into shuffled batches of size batch_size.
    Returns a list of numpy.ndarray, each containing indices for one batch.
    """
    idx = np.arange(num)
    np.random.shuffle(idx)
    splits = []
    cur = 0
    remaining = num
    while remaining > batch_size:
        remaining -= batch_size
        if remaining != 0:
            splits.append(cur + batch_size)
            cur += batch_size
    return np.split(idx, splits)


# Example of applying initializers to a simple nn.Linear layer
if __name__ == "__main__":
    # sample layer
    layer = nn.Linear(128, 64)
    weight_initializer(layer.weight, std=0.1)
    bias_initializer(layer.bias, val=0.01)

    # test get_shuffle_idx
    batches = get_shuffle_idx(100, 32)
    for i, batch in enumerate(batches):
        print(f"Batch {i}: size {len(batch)}")
