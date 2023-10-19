""" Algorithm for heuristic search for compositional explanations. """

from collections import Counter
import queue as Q

import torch

from src import formula as F
from src import mask_utils
from src import heuristics
from src import utils
from src import metrics


def compute_next_search_space(formulas, candidate_labels):
    """Compute the next search space starting from the current beam
    of formulas.

    Args:
        formulas (list): A list of formulas.
        candidate_labels (list): A list of candidate labels.

    Returns:
        search_space (list): A list of formulas.
    """
    search_space = []

    for formula in formulas:
        vals_formula = set(formula.get_vals())
        for candidate_term in candidate_labels:
            for op, negate in [(F.Or, False),
                               (F.And, False),
                               (F.And, True)
                               ]:
                if negate:
                    candidate_term = F.Not(candidate_term)
                candidate_formula = op(formula, candidate_term)
                candidate_formula.iou = 1.0

                # remove dummy cases with void masks or equivalent formulas
                if candidate_term.val in vals_formula and \
                        len(candidate_formula) <= 3:
                    continue
                search_space.append(candidate_formula)
    search_space = list(set(search_space))
    return search_space


def beam_search(
    search_space,
    *,
    masks,
    beam_masks,
    bitmaps,
    beam_limit,
    previous_beam=None,
):
    """Perform the beam search on the search space.

    Args:
        search_space (list): A list of formulas.
        masks (dict): A dictionary of concept masks. Each mask is a tensor of
            shape (N, H, W).
        beam_masks (dict): A dictionary of labal masks of the formulas in the
        current beam. Each mask is a tensor of shape (N, H, W).
        bitmaps (torch.Tensor): A tensor of shape (N, H, W) where N is the
            number of sample.
        beam_limit (int): The beam size.
        previous_beam (dict): A dictionary of the beam formulas and their iou.

    Returns:
        current_beam_formulas (list): A list of formulas.
        current_beam_iou (list): A list of iou.
        visited_indices (int): The number of visited indices.
    """
    if previous_beam is None:
        previous_beam = {}
    current_beam = Q.PriorityQueue(beam_limit)
    current_beam_iou = []
    current_beam_formulas = []
    minimum = 0
    visited_indices = 0
    best_formula = None
    # Init beam with previous best
    for k, v in previous_beam.items():
        if not current_beam.full():
            k.iou = v
            candidate = F.OrderedFormula(k)
            candidate.iou = k.iou
            current_beam.put(candidate)
            minimum = current_beam.queue[0].iou
        elif v > minimum:
            current_beam.get()
            candidate = F.OrderedFormula(v)
            candidate.iou = v.iou
            current_beam.put(candidate)
            minimum = current_beam.queue[0].iou
    if current_beam.empty():
        minimum = 0
    else:
        minimum = current_beam.queue[0].iou

    for candidate in search_space:
        e_iou = candidate.iou

        if current_beam.full() and e_iou < minimum:
            break
        candidate_formula = candidate.formula
        # skip equivalent formulas of the current beam
        if best_formula and hash(candidate_formula) == hash(best_formula):
            continue

        masks_formula = mask_utils.get_formula_mask(
            candidate_formula, masks, beam_masks
        ).to(bitmaps.device)
        iou = metrics.iou(
            masks_formula, bitmaps
        )
        visited_indices += 1

        if not current_beam.full():
            candidate.iou = iou
            current_beam.put(candidate)
            minimum = current_beam.queue[0].iou
        elif iou > minimum:
            candidate.iou = iou
            current_beam.get()
            current_beam.put(candidate)
            minimum = current_beam.queue[0].iou

    for _ in range(current_beam.qsize()):
        candidate = current_beam.get()
        current_beam_formulas.append(candidate.formula)
        current_beam_iou.append(candidate.iou)
    return current_beam_formulas, current_beam_iou, visited_indices


def get_beam_info(heuristic, beam, masks, bitmaps, mask_shape, device):
    """Compute the heuristic info for the beam.

    Args:
        heuristic (str): The heuristic to use.
        beam (dict): A dictionary of formulas of the current beam.
        masks (dict): A dictionary of concept masks. Each mask is a tensor of
            shape (N, H, W).
        bitmaps (torch.Tensor): A tensor of shape (N, H, W) where N is the
            number of sample.
        mask_shape (tuple): The shape of the mask.
        device (torch.device): The device to use.

    Returns:
        beam_masks (dict): A dictionary of labal masks of the formulas in the
        current beam. Each mask is a tensor of shape (N, H, W).
        updated_info (dict): A dictionary of heuristic info.
    """

    # update infos
    beam_masks = {}
    areas = {}
    intersections = {}
    inscribed = {}
    rectangles = {}
    for formula in beam:
        if isinstance(formula, F.Leaf):
            # Skip leaf. We have already them in masks
            continue
        # Compute formula mask
        masks_formula = mask_utils.get_formula_mask(formula, masks)
        mask_type = (
            "tensor" if isinstance(masks_formula, torch.Tensor) else "sparse"
        )
        if mask_type == "torch":
            beam_masks[formula] = masks_formula
        elif mask_type == "sparse":
            beam_masks[formula] = utils.torch_to_sparse(masks_formula)
        masks_formula = masks_formula.to(bitmaps.device)
        # Compute heuristic info
        if heuristic == "areas":
            areas[formula] = masks_formula.sum(1, dtype=torch.int32)
        elif heuristic == "cfh":
            areas[formula] = masks_formula.sum(1, dtype=torch.int32)
            intersections[formula] = (masks_formula & bitmaps).sum(
                1, dtype=torch.int32
            )
            masks_formula = masks_formula.reshape(
                -1, mask_shape[0], mask_shape[1]
            )
        elif heuristic == "mmesh":
            areas[formula] = masks_formula.sum(1, dtype=torch.int32)
            intersections[formula] = (masks_formula & bitmaps).sum(
                1, dtype=torch.int32
            )
            masks_formula = masks_formula.reshape(
                -1, mask_shape[0], mask_shape[1]
            )
            inscribed[formula] = mask_utils.get_inscribed_rectangles(
                masks_formula, device
            ).to(device)
            rectangles[formula] = mask_utils.get_overscribed_rectangles(
                masks_formula, mask_shape
            ).to(device)
    if heuristic == "areas":
        return beam_masks, areas
    elif heuristic == "mmesh":
        return beam_masks, (areas, inscribed, rectangles, intersections)
    elif heuristic == "cfh":
        return beam_masks, (areas, intersections)


def perform_heuristic_search(
    heuristic,
    netdissect_rank,
    masks,
    bitmaps,
    heuristic_info,
    num_hits,
    *,
    max_size_mask,
    beam_size=5,
    length=3,
    mask_shape=None,
    device=torch.device("cpu"),
):
    """Compute the heuristic score for each concept in the candidate_concepts
    list for the given bitmaps.

    Args:
        bitmaps (torch.Tensor): A tensor of shape (N, H, W) where N is the
            number of sample.
        masks (dict): A dictionary of concept masks. Each mask is a tensor of
            shape (H, W).
        candidate_concepts (list): A list of candidate concepts.

    Returns:
        heuristic_rank (dict): A dictionary of concept scores. Each score is
            a float.
    """
    # Extract first beam and candidate concepts
    netdissect_rank = Counter(netdissect_rank)
    beam = {
        F.Leaf(lab): iou
        for lab, iou in netdissect_rank.most_common(beam_size * 2)
        if iou > 0
    }
    candidate_labels = [F.Leaf(lab) for lab in range(1, len(masks))]
    # Beam Search
    total_visited = 0
    beam_masks = {}
    updated_info = {}
    for index_loop in range(1, length):
        # update infos
        if heuristic != "none":
            beam_masks, updated_info = get_beam_info(
                heuristic, beam, masks, bitmaps, mask_shape, device
            )
        sorted_search_space = heuristics.sort_search_space_by(
            compute_next_search_space(
                beam.keys(), candidate_labels
            ),
            heuristic,
            heuristic_info=(heuristic_info, updated_info),
            num_hits=num_hits,
            max_size_mask=max_size_mask,
        )
        if index_loop == length - 1:
            beam_size = 1
        next_beam_formulas, next_beam_iou, beam_visited = beam_search(
            sorted_search_space,
            masks=masks,
            previous_beam=beam,
            beam_masks=beam_masks,
            bitmaps=bitmaps,
            beam_limit=beam_size,
        )
        total_visited = total_visited + beam_visited
        # Update top formulas
        for index_beam in range(len(next_beam_formulas)):
            beam.update(
                {next_beam_formulas[index_beam]: next_beam_iou[index_beam]}
            )

        # Trim the beam
        beam = dict(Counter(beam).most_common(beam_size))
    top_result = Counter(beam).most_common(1)[0]

    best_iou = top_result[1].item()
    best_label = top_result[0]
    return best_label, best_iou, total_visited
