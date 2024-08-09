"""
Utility functions.
"""
import os
import random

import torch
import numpy as np
from numpy.random import RandomState
import scipy.sparse as sparse


def set_seed(seed: int) -> RandomState:
    """Method to set seed across runs to ensure reproducibility.
    It fixes seed for single-gpu machines.
    Args:
        seed (int): Seed to fix reproducibility. It should different for
            each run
    Returns:
        RandomState: fixed random state to initialize dataset iterators
    """
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = (
        False  # set to false for reproducibility, True to boost performance
    )
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    # os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:2"
    g = torch.Generator()
    g.manual_seed(0)
    return g


# reference: https://pytorch.org/docs/stable/notes/randomness.html
def seed_worker(worker_id):
    """Method to set seed for each worker.
    Args:
        worker_id (int): Id of the worker
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def torch_to_sparse(vector):
    """
    Convert a torch tensor to a sparse matrix.

    Args:
        vector (torch.Tensor): tensor to convert

    Returns:
        scipy.sparse.csr_matrix: sparse matrix
    """
    if len(vector.shape) != 2:
        vector = torch.reshape(vector, (vector.shape[0], -1))
    return sparse.csr_matrix(vector.numpy())


def sparse_to_torch(vector):
    """
    Convert a sparse matrix to a torch tensor.

    Args:
        vector (scipy.sparse.csr_matrix): sparse matrix to convert

    Returns:
        torch.Tensor: tensor
    """
    return torch.from_numpy(vector.toarray())
