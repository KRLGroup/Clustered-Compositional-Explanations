"""Metrics to evaluate the quality of the explanations."""

import functools

import torch

from src import model_utils
from src import constants as C


@functools.lru_cache(maxsize=10)
def compute_hits(vector):
    """Compute the number of ones in the given vector.
    Args:
        vector (torch.Tensor): A tensor of shape (N, H, W) where N is the
            number of sample.
    Returns:
        hits (int): The number of ones in the given vector.
    """
    return torch.count_nonzero(vector)


def iou(vector1, vector2):
    """Compute the intersection over union between two vectors.
    Args:
        vector1 (torch.Tensor): A tensor of shape (N, H, W) where N is the
            number of sample.
        vector2 (torch.Tensor): A tensor of shape (N, H, W) where N is the
            number of sample.
    Returns:
        iou (float): The intersection over union between the two vectors.
    """
    intersection = torch.count_nonzero(vector1 & vector2)
    v1_size = compute_hits(vector1)
    v2_size = compute_hits(vector2)
    score = intersection / max(v1_size + v2_size - intersection, C.EPSILON)
    return score


def sample_iou(vector1, vector2):
    """Compute the intersection over union between two vectors.
    Args:
        vector1 (torch.Tensor): A tensor of shape (N, H, W) where N is the
            number of sample.
        vector2 (torch.Tensor): A tensor of shape (N, H, W) where N is the
            number of sample.
    Returns:
        iou (float): The intersection over union between the two vectors.
    """
    intersection = torch.count_nonzero(vector1 & vector2, 1)
    v1_size = torch.count_nonzero(vector1, 1)
    v2_size = torch.count_nonzero(vector2, 1)
    score = intersection / (v1_size + v2_size - intersection + C.EPSILON)
    return score


def activations_coverage(activations, segmentations):
    """Compute the activation coverage for the given activations and
    segmentations.
    Args:
        activations (torch.Tensor): A tensor of shape (N, H, W) where N is
            the number of sample.
        segmentations (torch.Tensor): A tensor of shape (N, H, W) where N is
            the number of sample.
    Returns:
        activation_coverage (float): The activation coverage.
    """
    return torch.count_nonzero(
        activations & segmentations
    ) / torch.count_nonzero(activations)


def detection_accuracy(activations, segmentations):
    """Compute the segmentations coverage for the given activations and
    segmentations.
    Args:
        activations (torch.Tensor): A tensor of shape (N, H, W) where N is the
            number of sample.
        segmentations (torch.Tensor): A tensor of shape (N, H, W) where N is
            the number of sample.
    Returns:
        segmentations_coverage (float): The segmentations coverage.
    """
    return torch.count_nonzero(
        activations & segmentations
    ) / torch.count_nonzero(segmentations)


def samples_coverage(activations, segmentations):
    """Compute the samples coverage for the given activations and
    segmentations.
    Args:
        activations (torch.Tensor): A tensor of shape (N, H, W) where N is
            the number of sample.
        segmentations (torch.Tensor): A tensor of shape (N, H, W) where N is
        the number of sample.
    Returns:
        samples_coverage (float): The samples coverage.
    """
    samples_overlap = (
        torch.sum(activations & segmentations, 1, dtype=torch.int32) > 0
    )
    segmentation_in = torch.sum(segmentations, 1, dtype=torch.int32) > 0
    return torch.sum(samples_overlap) / torch.sum(segmentation_in)


def avg_mask_size(mask):
    """Compute the average mask size for the given mask considering
    only the samples where the mask has at least one pixel.
    Args:
        mask (torch.Tensor): A tensor of shape (N, F) where N is
            the number of sample.
    Returns:
        avg_size (float): The average segmentation size.
    """
    assert len(mask.shape) == 2
    samples_size = torch.sum(mask, 1, dtype=torch.int32)
    samples_size = samples_size[samples_size > 0]
    samples_size = samples_size / mask.shape[1]
    return torch.mean(samples_size.float())


def avg_overlapping(activations, segmentations):
    """Compute the average overlapping between the given activations and
    segmentations by considering only the samples where the intersection is
    greater than zero.
    Args:
        activations (torch.Tensor): A tensor of shape (N, F) where N is the
            number of sample.
        segmentations (torch.Tensor): A tensor of shape (N, F) where N is the
            number of sample.
    Returns:
        avg_overlapping (float): The average overlapping.
    """
    overlapping = torch.sum(activations & segmentations, 1, dtype=torch.int32)
    assert len(activations.shape) == 2

    overlapping = overlapping[overlapping > 0]
    overlapping = overlapping / activations.shape[1]
    return torch.mean(overlapping)


def get_num_nonzerosamples(mask):
    """Compute the number of samples with at least one pixel.
    Args:
        mask (torch.Tensor): A tensor of shape (N, H, W) where N is the
            number of sample.
    Returns:
        num_nonzerosamples (int): The number of samples with at
            least one pixel.
    """
    return torch.sum(torch.sum(mask, 1, dtype=torch.int32) > 0)


def cosine_concept_masking_score(
        activation_before_masking, activation_after_masking, activation_range):
    """Compute the concept masking for the given activations.
    Args:
        activation_before_masking (torch.Tensor): A tensor of shape (N, H, W)
            where N is the number of sample.
        activation_after_masking (torch.Tensor): A tensor of shape (N, H, W)
            where N is the number of sample.
    Returns:
        concept_masking (float): The concept masking.
    """

    activation_before_masking = activation_before_masking.flatten(1)
    bitmaps = torch.where(
                    (activation_before_masking > activation_range[0])
                    & (activation_before_masking < activation_range[1]),
                    1, 0)
    activation_after_masking = activation_after_masking.flatten(1)
    activation_before_masking = activation_before_masking * bitmaps
    activation_after_masking = activation_after_masking * bitmaps
    return torch.mean(torch.nn.functional.cosine_similarity(
        activation_before_masking, activation_after_masking
    ))


def get_concept_masking(
    activations,
    mask_shape,
    label_mask,
    loader,
    model,
    layer_name,
    unit,
    input_size,
    activation_range,
):
    """Compute the concept masking for the given label.
    Args:
        label (torch.Tensor):
    Returns:
        concept_masking (float):
    """
    samples_with_label = torch.nonzero(label_mask.sum(1) > 0).flatten()
    label_mask = label_mask.reshape(-1, mask_shape[0], mask_shape[1])
    no_mask_activations = activations[samples_with_label].unsqueeze(1)
    random_masks = torch.rand(label_mask.shape, device=label_mask.device)
    mask_to_apply = torch.where(label_mask == 0, random_masks, label_mask)
    masked_activations = model_utils.apply_concept_masking(
        loader,
        model,
        [layer_name],
        units=[unit],
        mask=mask_to_apply,
        image_size=input_size,
    )[0][samples_with_label]
    score = cosine_concept_masking_score(
        no_mask_activations, masked_activations, activation_range)
    return score
