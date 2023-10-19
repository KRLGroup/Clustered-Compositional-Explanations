"""Activation utils.

This module contains functions to compute activation ranges and
bitmaps of activations.
"""

from typing import List, Tuple

import torch
import sklearn.cluster as scikit_cluster

from src import vecquantile
from src import constants as C


def build_ranges_from_clusters(
        activations: torch.Tensor, clusters: List[int],
        num_clusters: int) -> List[tuple]:
    """Build activation ranges from clusters.

    Args:
        activations (torch.Tensor): Activations of the unit.
        clusters (List[int]): Clusters indexes of the activations.
        num_clusters (int): Number of clusters.

    Returns:
        activation_ranges (List[tuple]): Activation ranges for each cluster.
    """

    activations_ranges = []
    for label in range(num_clusters):
        cluster_activations = activations[clusters == label]
        lower_bound = torch.min(cluster_activations)
        upper_bound = torch.max(cluster_activations)
        activations_ranges.append((lower_bound.item(), upper_bound.item()))
    return activations_ranges


def compute_activation_ranges(
        activations: torch.Tensor, num_clusters: int) -> List[Tuple]:
    """Compute activation ranges for each unit.

    Args:
        activations (torch.Tensor): Activations of the unit.
        num_clusters (int): Number of clusters.
        algorithm (str): Algorithm to use for clustering.

    Returns:
        activation_ranges (List[tuple]): Activation ranges for each unit.
    """
    if num_clusters == 1:
        # Case vanilla compositional and netdissect range
        # Avoid zero is set to false like in the compositional paper
        threshold = quantile_threshold(
            activations, quantile=C.NETDISSECT_QUANTILE, avoid_zero=False
        )
        activation_ranges = [(threshold, torch.tensor(float("inf")))]
    else:
        activations = activations.reshape(-1, 1)
        # Remove zeros from activations if there is a relu activation
        if torch.all(activations >= 0):
            activations = activations[activations > 0]
            activations = activations.reshape(-1, 1)
        # Compute activation ranges
        clusters = scikit_cluster.KMeans(
            n_clusters=num_clusters, random_state=0
            ).fit(activations)
        activation_ranges = build_ranges_from_clusters(
            activations, clusters.labels_, num_clusters)
    return activation_ranges


def compute_bitmaps(
        activations: torch.Tensor, activation_range: Tuple,
        mask_shape: List[int]) -> torch.Tensor:
    """Get the bitmaps of the unit.

    This function upsamples the activations to the original size of the
    image and then binarize them.
    Args:
        activations (torch.Tensor): Activations of the unit.
        activation_range (Tuple): Activation range of the unit.
        mask_shape (List[int]): Shape of the mask.

    Returns:
        bitmaps (torch.Tensor): Bitmaps of the unit.
    """
    lower, upper = activation_range
    upsampled_activations = torch.nn.functional.interpolate(
        activations.unsqueeze(1),
        size=mask_shape, mode='bilinear')
    upsampled_activations = upsampled_activations.squeeze(1)
    bitmaps = torch.where(
        (upsampled_activations > lower) & (upsampled_activations < upper),
        True, False)
    bitmaps = bitmaps.reshape(bitmaps.shape[0], -1)
    return bitmaps


def quantile_threshold(
        layer_activations: torch.Tensor, quantile: float, *,
        avoid_zero: bool, batch_size=64, seed=1) -> torch.Tensor:
    """
    Determine thresholds for neuron activations for each neuron.

    Args:
        layer_activations (torch.Tensor): Activations of the layer.
        quantile (float): Quantile to use.
        avoid_zero (bool): Whether to remove zeros from the activations.
        batch_size (int): Batch size to use.
        seed (int): Seed to use for the quantile vector.

    Returns:
        thresholds (torch.Tensor): Thresholds for each neuron.
    """
    quant = vecquantile.QuantileVector(depth=1, seed=seed)
    for i in range(0, layer_activations.shape[0], batch_size):
        batch = layer_activations[i:i + batch_size]
        batch = batch.flatten().reshape(-1, 1)
        if avoid_zero:
            batch = batch[batch != 0].reshape(-1, 1)
        quant.add(batch)
    thresholds = quant.readout(1000)[:, int(1000 * (1 - quantile) - 1)]
    return torch.tensor(thresholds)
