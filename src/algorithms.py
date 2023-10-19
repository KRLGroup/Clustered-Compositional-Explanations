"""
This module contains the implementation of the algorithms described in
the paper. It includes NetDissect, the compositional explanations algorithm
and the heuristic search.
"""
from collections import Counter
import torch

from src import heuristic_search
from src import utils
from src import metrics


def get_netdissect_scores(bitmaps, masks):
    """Compute the NetDissect score for each concept in the candidate_concepts
    list for the given bitmaps.

    Args:
        bitmaps (torch.Tensor): A tensor of shape (N, H, W) where N is the
            number of sample.
        masks (dict): A dictionary of concept masks. Each mask is a tensor of
            shape (H, W).
        candidate_concepts (list): A list of candidate concepts.

    Returns:
        netdissect_rank (dict): A dictionary of concept scores. Each score is
            a float.
    """
    # hits_unit = torch.count_nonzero(bitmaps)
    netdissect_rank = {}
    mask_type = "tensor" if isinstance(masks[1], torch.Tensor) else "sparse"
    candidate_concepts = range(1, len(masks))
    for concept in candidate_concepts:
        if mask_type == "tensor":
            concept_mask = masks[concept]
        else:
            concept_mask = utils.sparse_to_torch(masks[concept])
        concept_mask = concept_mask.to(bitmaps.device)
        concept_iou = metrics.iou(concept_mask, bitmaps)
        netdissect_rank[concept] = concept_iou.item()

    return netdissect_rank


def get_augmented_netdissect_scores(bitmaps, masks):
    """Compute the NetDissect score for each concept in the candidate_concepts
    list for the given bitmaps.

    Args:
        bitmaps (torch.Tensor): A tensor of shape (N, H, W) where N is the
            number of sample.
        masks (dict): A dictionary of concept masks. Each mask is a tensor of
            shape (H, W).
        candidate_concepts (list): A list of candidate concepts.

    Returns:
        netdissect_rank (dict): A dictionary of concept scores. Each score is
            a float.
    """
    netdissect_rank = {}
    areas = [None]
    mask_type = "tensor" if isinstance(masks[1], torch.Tensor) else "sparse"
    candidate_concepts = range(1, len(masks))
    for concept in candidate_concepts:
        if mask_type == "tensor":
            concept_mask = masks[concept]
        else:
            concept_mask = utils.sparse_to_torch(masks[concept])
        concept_mask = concept_mask.to(bitmaps.device)
        concept_iou = metrics.iou(concept_mask, bitmaps)
        intersection_area = (concept_mask & bitmaps).sum(
            dim=1, dtype=torch.int32
        )
        netdissect_rank[concept] = concept_iou
        areas.append(intersection_area)
    return netdissect_rank, areas


def get_heuristic_scores(
    segmentations,
    activation_masks,
    *,
    heuristic="mmesh",
    segmentations_info=None,
    max_size_mask,
    beam_size=5,
    length=3,
    mask_shape=None,
    device=torch.device("cpu")
):
    """Compute the heuristic score for each concept in the candidate_concepts
    list for the given bitmaps.

    Args:
        segmentations (dict): A dictionary of concept masks. Each mask is a
            tensor of shape (N, H, W) where N is
            the number of sample.
        activation_masks (torch.Tensor): A tensor of shape (N, H, W) where N is
            the number of sample.
        heuristic (str): The heuristic to use for the search. Can be one of
            "mmesh", "cfh", "areas", "none".
        segmentations_info (dict): A dictionary of information about the
            segmentations. None can be used only when the heuristic is none.
        max_size_mask (int): The maximum size of the masks.
        beam_size (int): The beam size for the search.
        length (int): The length of the search.
        mask_shape (tuple): The shape of the masks.
        device (torch.device): The device to use for the computation.

    Returns:
        best_label (int): The label of the best concept.
        best_iou (float): The IOU of the best concept.
        visited (int): The number of visited nodes.
    """

    if segmentations_info is None and heuristic != "none":
        raise ValueError(
            "segmentations_info must be provided when heuristic is not none"
        )
    # Compute commong parameters
    num_hits = activation_masks.sum()

    if length == 1:
        # vanilla netdissect
        rank = get_netdissect_scores(activation_masks, segmentations)
        best_label = Counter(rank).most_common(1)[0][0]
        best_iou = Counter(rank).most_common(1)[0][1]
        return best_label, best_iou, 0
    if heuristic == "areas":
        sample_activation_areas = activation_masks.sum(1)
        netdissect_scores = get_netdissect_scores(
            activation_masks, segmentations
        )
        heuristic_info = (segmentations_info, sample_activation_areas)
    elif heuristic == "mmesh":
        sample_activation_areas = activation_masks.sum(1)
        netdissect_scores, intersect_areas = get_augmented_netdissect_scores(
            activation_masks, segmentations
        )
        heuristic_info = (
            segmentations_info,
            sample_activation_areas,
            intersect_areas,
        )
    elif heuristic == "cfh":
        sample_activation_areas = activation_masks.sum(1)
        netdissect_scores, intersect_areas = get_augmented_netdissect_scores(
            activation_masks, segmentations
        )
        heuristic_info = (
            segmentations_info,
            sample_activation_areas,
            intersect_areas,
        )
    elif heuristic == "none":
        netdissect_scores = get_netdissect_scores(
            activation_masks, segmentations
        )
        heuristic_info = None
    best_label, best_iou, visited = heuristic_search.perform_heuristic_search(
        heuristic,
        netdissect_scores,
        segmentations,
        activation_masks,
        heuristic_info,
        num_hits,
        beam_size=beam_size,
        length=length,
        max_size_mask=max_size_mask,
        mask_shape=mask_shape,
        device=device,
    )
    return best_label, best_iou, visited
