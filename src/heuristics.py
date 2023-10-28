import torch

from src import formula as F


def formula_is_in(f, unary_areas, max_mask_size, enneary_areas=None):
    """
    Function to check where the formula is present in the data.

    Args:
        unary_areas (list): list of areas of the masks.
        f (src.formula.Formula): formula to check.
        max_mask_size (int): maximum size of the mask.
        enneary_areas (dict): dictionary of additional areas.

    Returns:
        Boolean list
    """
    if enneary_areas is not None and f in enneary_areas.keys():
        return enneary_areas[f] > 0
    if isinstance(f, F.And):
        masks_l = formula_is_in(
            f.left, unary_areas, max_mask_size, enneary_areas)
        masks_r = formula_is_in(
            f.right, unary_areas, max_mask_size, enneary_areas)
        return masks_l & masks_r
    elif isinstance(f, F.Or):
        masks_l = formula_is_in(
            f.left, unary_areas, max_mask_size, enneary_areas)
        masks_r = formula_is_in(
            f.right, unary_areas, max_mask_size, enneary_areas)
        return masks_l | masks_r
    elif isinstance(f, F.Not):
        return unary_areas[f.val.val] < max_mask_size
    elif isinstance(f, F.Leaf):
        return unary_areas[f.val] > 0


def get_coordinates(term, leaf_list, enneary_list):
    """Returns the coordinates of  of the formula
    Args:
        term (src.formula.Formula): term to check.
        leaf_list (dict): dictionary of leaf coordinates.
        enneary_list (dict): dictionary of enneary formulas coordinates.

    Returns:
        Coordinates of the formula
    """
    if isinstance(term, F.BinaryNode):
        coordinates = enneary_list[term]
    elif isinstance(term, F.Leaf):
        coordinates = leaf_list[term.val]
    else:
        coordinates = term
    return coordinates


def get_rectangles_overlap(coordinates_a, coordinates_b, return_points=False):
    """Returns the overlap between two rectangles given their
        top left and bottom right coordinates"""
    a_top_left_x = coordinates_a[:, 0, 1]
    a_top_left_y = coordinates_a[:, 0, 0]
    a_bottom_right_x = coordinates_a[:, 1, 1] + 1
    a_bottom_right_y = coordinates_a[:, 1, 0] + 1
    b_top_left_x = coordinates_b[:, 0, 1]
    b_top_left_y = coordinates_b[:, 0, 0]
    b_bottom_right_x = coordinates_b[:, 1, 1] + 1
    b_bottom_right_y = coordinates_b[:, 1, 0] + 1
    x_overlap = torch.maximum(
        torch.zeros_like(a_bottom_right_x),
        torch.minimum(a_bottom_right_x, b_bottom_right_x)
        - torch.maximum(a_top_left_x, b_top_left_x),
    )
    y_overlap = torch.maximum(
        torch.zeros_like(a_bottom_right_y),
        torch.minimum(a_bottom_right_y, b_bottom_right_y)
        - torch.maximum(a_top_left_y, b_top_left_y),
    )
    overlap = x_overlap * y_overlap
    if return_points:
        # compute coordinates of the intersection rectangle
        top_left_x = torch.maximum(a_top_left_x, b_top_left_x)
        top_left_y = torch.maximum(a_top_left_y, b_top_left_y)
        bottom_right_x = torch.minimum(a_bottom_right_x, b_bottom_right_x) - 1
        bottom_right_y = torch.minimum(a_bottom_right_y, b_bottom_right_y) - 1
        return overlap, torch.tensor(
            list(
                zip(
                    torch.tensor(list(zip(top_left_y, top_left_x))),
                    torch.tensor(list(zip(bottom_right_y, bottom_right_x))),
                )
            )
        )
    else:
        return overlap


def get_intersection_info(
    term, unary_intersect, enneary_intersect, neuron_areas
):
    """
    Returns the intersection between the term and the firing areas
    Args:
        term (src.formula.Formula): term to check.
        unary_intersect (dict): dictionary of unary intersections.
        enneary_intersect (dict): dictionary of enneary intersections.
        neuron_areas (torch.tensor): tensor of neuron areas.

    Returns:
        Intersection between the term and the firing areas
    """

    if isinstance(term, F.BinaryNode):
        term_and_fires_areas = enneary_intersect[term]
    elif isinstance(term, F.Not):
        term_and_fires_areas = neuron_areas - unary_intersect[term.val.val]
    else:
        term_and_fires_areas = unary_intersect[term.val]

    return term_and_fires_areas


def get_area_info(term, unary_areas, enneary_areas, max_size_mask):
    """
    Returns the area of the term
    Args:
        term (src.formula.Formula): term to check.
        unary_areas (dict): dictionary of unary areas.
        enneary_areas (dict): dictionary of enneary areas.
        max_size_mask (int): maximum size of the mask.

    Returns:
        Area of the term
    """
    if isinstance(term, F.BinaryNode):
        areas = enneary_areas[term]
    elif isinstance(term, F.Not):
        areas = max_size_mask - get_area_info(
            term.val, unary_areas, enneary_areas, max_size_mask
        )
    else:
        areas = unary_areas[term.val]
    return areas


def is_scene(areas, max_size_mask):
    """Returns True if the mask is a scene, False otherwise"""
    condition = (areas == 0) | (areas == max_size_mask)
    if condition.all():
        flag = True
    else:
        flag = False
    return flag


def compute_scene_iou(
    formula,
    left_areas,
    right_areas,
    left_intersection_area,
    right_intersection_area,
    neuron_areas,
    max_size_mask,
    num_hits
):
    """Computes the IoU of a scene formula
    Args:
        formula (src.formula.Formula): formula to check.
        left_areas (torch.tensor): sample areas of the left term.
        right_areas (torch.tensor): sample areas of the right term.
        left_intersection_area (torch.tensor): intersection areas
            with the neuron of the left term.
        right_intersection_area (torch.tensor): intersection areas
            with the neuron of the right term.
        neuron_areas (torch.tensor): tensor of neuron areas.
        max_size_mask (int): maximum size of the mask.
        num_hits (int): number of hits.
    Returns:
        IoU of the scene formula
    """
    if isinstance(formula, F.Or):
        # exact computation
        formula_mask = torch.where(
            left_areas > right_areas, left_areas, right_areas
        )
        intersection = torch.where(
            formula_mask == max_size_mask,
            neuron_areas,
            left_intersection_area + right_intersection_area,
        )

        intersection = torch.sum(intersection)
    elif isinstance(formula, F.And):
        formula_mask = torch.minimum(left_areas, right_areas)
        intersection = torch.where(
            left_intersection_area < right_intersection_area,
            left_intersection_area,
            right_intersection_area,
        )
        intersection = torch.sum(intersection)
    estimated_iou = intersection / (
        num_hits + torch.sum(formula_mask) - intersection
    )
    return torch.round(estimated_iou, decimals=4)


def mmesh_heuristic(formula, heuristic_info, *, num_hits, max_size_mask):
    """
    Computes the IoU of a formula using the mmesh heuristic.
    Args:
        formula (src.formula.Formula): formula to check.
        heuristic_info (tuple): tuple of unary and enneary heuristic_info collected from
            the dataset and the parsing of the previous beam.
        num_hits (int): number of hits in the neuron's mask.
        max_size_mask (int): maximum size of the mask.
    Returns:
        float: estimated iou
    """
    dissect_info = heuristic_info[0]
    enneary_info = heuristic_info[1]
    unary_info, neuron_areas, unary_intersection = dissect_info
    unary_areas, (unary_inscribed, unary_bounding_box) = unary_info
    enneary_areas, enneary_inscribed, enneary_bounding_box, enneary_intersection = enneary_info

    formula_in = formula_is_in(
         formula, unary_areas, max_size_mask, enneary_areas
    )

    left_and_fires_areas = get_intersection_info(
        formula.left, unary_intersection, enneary_intersection, neuron_areas
    )
    right_and_fires_areas = get_intersection_info(
        formula.right, unary_intersection, enneary_intersection, neuron_areas
    )
    left_areas = get_area_info(
        formula.left, unary_areas, enneary_areas, max_size_mask
    )
    right_areas = get_area_info(
        formula.right, unary_areas, enneary_areas, max_size_mask
    )

    left_intersection_area = left_and_fires_areas * formula_in
    right_intersection_area = right_and_fires_areas * formula_in

    # In case of scene formula, we can compute the exact formula mask
    # and in the OR case, we can compute the exact intersection
    left_is_scene = is_scene(left_areas, max_size_mask)
    right_is_scene = is_scene(right_areas, max_size_mask)
    if left_is_scene or right_is_scene:
        return compute_scene_iou(
            formula,
            left_areas,
            right_areas,
            left_intersection_area,
            right_intersection_area,
            neuron_areas,
            max_size_mask,
            num_hits,
        )

    # Otherswise, we have to approximate both of them
    if isinstance(formula, F.Or):
        max_intersection_neuron = torch.minimum(
            neuron_areas, left_intersection_area + right_intersection_area
        )
        minimum_area_mask = torch.maximum(left_areas, right_areas)
        coordinates_left = get_coordinates(
            formula.left, unary_bounding_box, enneary_bounding_box
        )
        coordinates_right = get_coordinates(
            formula.right, unary_bounding_box, enneary_bounding_box
        )
        maximum_intersection = get_rectangles_overlap(
            coordinates_left, coordinates_right
        )
        minimum_area_mask = torch.maximum(
            minimum_area_mask,
            left_areas + right_areas - maximum_intersection,
        )
        minimum_area_mask = torch.maximum(
            minimum_area_mask, max_intersection_neuron
        )
    elif isinstance(formula, F.And):
        max_intersection_neuron = torch.minimum(
            left_intersection_area, right_intersection_area
        )
        if isinstance(formula.right, F.Not):
            coordinates_left = get_coordinates(
                formula.left, unary_bounding_box, enneary_bounding_box
            )
            coordinates_right = get_coordinates(
                formula.right.val, unary_bounding_box, enneary_bounding_box
            )
            maximum_intersection = get_rectangles_overlap(
                coordinates_left, coordinates_right
            )
            minimum_area_mask = left_areas - maximum_intersection

        else:
            coordinates_left = get_coordinates(
                formula.left, unary_inscribed, enneary_inscribed
            )
            coordinates_right = get_coordinates(
                formula.right, unary_inscribed, enneary_inscribed
            )
            minimum_area_mask = get_rectangles_overlap(
                coordinates_left, coordinates_right
            )
        minimum_area_mask = torch.maximum(
            minimum_area_mask, max_intersection_neuron
        )

    max_intersection_neuron = torch.sum(max_intersection_neuron)
    minimum_area_mask = torch.sum(minimum_area_mask)
    estimated_iou = max_intersection_neuron / (
        num_hits + minimum_area_mask - max_intersection_neuron
    )
    return torch.round(estimated_iou, decimals=4)


def coordinates_free_heuristic(
        formula, heuristic_info, *, num_hits, max_size_mask):
    """ Heuristic that does not use coordinates
    to compute the minimum and maximum possible extension
    of the label mask.
    Args:
        formula (F.Formula): formula to estimate
        heuristic_info (tuple): tuple of information available for the heuristic
        num_hits (int): number of hits in the neuron mask
        max_size_mask (int): maximum size of the mask
    Returns:
        float: estimated iou
    """
    dissect_info = heuristic_info[0]
    enneary_info = heuristic_info[1]
    unary_areas, neuron_areas, unary_intersection = dissect_info
    enneary_areas, enneary_intersection = enneary_info

    formula_in = formula_is_in(
        formula, unary_areas, max_size_mask, enneary_areas
    )

    left_and_fires_areas = get_intersection_info(
        formula.left, unary_intersection, enneary_intersection, neuron_areas
    )
    right_and_fires_areas = get_intersection_info(
        formula.right, unary_intersection, enneary_intersection, neuron_areas
    )
    left_areas = get_area_info(
        formula.left, unary_areas, enneary_areas, max_size_mask
    )
    right_areas = get_area_info(
        formula.right, unary_areas, enneary_areas, max_size_mask
    )

    left_intersection_area = left_and_fires_areas * formula_in
    right_intersection_area = right_and_fires_areas * formula_in

    # In case of scene formula, we can compute the exact formula mask
    # and in the OR case, we can compute the exact intersection
    left_is_scene = is_scene(left_areas, max_size_mask)
    right_is_scene = is_scene(right_areas, max_size_mask)
    if left_is_scene or right_is_scene:
        return compute_scene_iou(
            formula,
            left_areas,
            right_areas,
            left_intersection_area,
            right_intersection_area,
            neuron_areas,
            max_size_mask,
            num_hits,
        )

    # Otherswise, we have to approximate both of them
    if isinstance(formula, F.Or):
        max_intersection_neuron = torch.minimum(
            neuron_areas, left_intersection_area + right_intersection_area
        )
    elif isinstance(formula, F.And):
        max_intersection_neuron = torch.minimum(
            left_intersection_area, right_intersection_area
        )

    max_intersection_neuron = torch.sum(max_intersection_neuron)
    estimated_iou = max_intersection_neuron / (
        num_hits - max_intersection_neuron
    )
    return torch.round(estimated_iou, decimals=4)


def areas_heuristic(formula, heuristic_info, *, num_hits, max_size_mask):
    """ Compute the heuristic based onnly on the areas of the formula
    and the areas of the neuron's mask.

    Args:
        formula (F.Formula): The formula to compute the heuristic for
        heuristic_info (tuple): The information to compute the heuristic
        num_hits (int): The number of hits in the neuron mask
        max_size_mask (int): The maximum size of the mask

    Returns:
        float: estimated iou
    """
    unary_info = heuristic_info[0]
    enneary_info = heuristic_info[1]
    unary_areas, neuron_areas = unary_info
    enneary_areas = enneary_info

    left_areas = get_area_info(
        formula.left, unary_areas, enneary_areas, max_size_mask
    )
    right_areas = get_area_info(
        formula.right, unary_areas, enneary_areas, max_size_mask
    )

    # Otherswise, we have to approximate both of them
    if isinstance(formula, F.Or):
        max_intersection_neuron = torch.minimum(
            neuron_areas, left_areas + right_areas
        )
    elif isinstance(formula, F.And):
        max_intersection_neuron = torch.minimum(left_areas, right_areas)

    max_intersection_neuron = torch.sum(max_intersection_neuron)
    estimated_iou = max_intersection_neuron / (
        num_hits - max_intersection_neuron
    )
    return torch.round(estimated_iou, decimals=4)


def sort_search_space_by(
        search_space, name_heuristic, *,
        heuristic_info, num_hits,  max_size_mask):
    """
    Sort the search space using the heuristic name_heuristic.

    Args:
        search_space (list of Formula): the search space to sort
        name_heuristic (str): the name of the heuristic to use
        heuristic_info (tuple): the information to be used by the heuristic
        num_hits (int): the number of hits of the neuron
        max_size_mask (int): the maximum size of the mask

    Returns:
        list of Formula: the sorted search space
    """

    if name_heuristic == "mmesh":
        heuristic = mmesh_heuristic
    elif name_heuristic == "areas":
        heuristic = areas_heuristic
    elif name_heuristic == "cfh":
        heuristic = coordinates_free_heuristic
    elif name_heuristic == "none":
        for index_formula, candidate_formula in enumerate(search_space):
            search_space[index_formula].iou = 1.0
            search_space[index_formula] = F.OrderedFormula(
                search_space[index_formula]
            )
        return search_space
    else:
        raise ValueError(f"Unknown heuristic: {name_heuristic}")

    for index_formula, candidate_formula in enumerate(search_space):
        esti = heuristic(
            candidate_formula, heuristic_info, num_hits=num_hits,
            max_size_mask=max_size_mask
        )
        search_space[index_formula] = F.OrderedFormula(
            search_space[index_formula]
        )
        search_space[index_formula].iou = esti

    search_space = sorted(search_space, reverse=True)

    return search_space
