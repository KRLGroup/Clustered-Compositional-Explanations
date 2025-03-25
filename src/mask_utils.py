""" Module to handle masks. """

from collections import defaultdict
import os
import pickle

import torch
from tqdm import tqdm
import scipy.sparse as sparse
import torchvision

from src import formula as F
from src import utils


def load_sparse_masks(concept_names, segmentations_directory):
    """
    Loads the sparse masks from the given directory.
    Args:
        concept_names (list): list of concept names.
        segmentations_directory (str): directory where the masks are stored.
    Returns:
        List of sparse masks.
    """

    masks_list = []
    masks_list.append(None)
    for concept in tqdm(
        concept_names, desc="Loading Sparse Masks", total=len(concept_names)
    ):
        if concept != "":
            masks_list.append(
                sparse.load_npz(f"{segmentations_directory}/{concept}.npz")
            )
    return masks_list


def save_sparse_masks(segloader, concept_names, segmentations_directory, device):
    """
    Saves the masks in a sparse format in the given directory.
    Args:
        segloader (torch.utils.data.DataLoader): dataloader
            for the segmentations.
        concept_names (list): list of concept names.
        segmentations_directory (str): directory where the masks are stored.
    """
    tot_concepts = len(concept_names)
    ranges = range(0, tot_concepts, 20)
    for starting_index in tqdm(ranges):
        concepts = range(starting_index, starting_index + 20)
        masks = defaultdict(lambda: [])
        for _, (_, segmentations, _) in enumerate(segloader):
            num_categories_segmentations = segmentations.shape[1]
            segmentations = segmentations.to(device)
            for concept_index in concepts:
                concept_mask = torch.zeros(
                    (
                        segmentations.shape[0],
                        segmentations.shape[2],
                        segmentations.shape[3],
                    ),
                    dtype=bool,
                ).to(device)

                for index_segmentation in range(num_categories_segmentations):
                    concept_mask = concept_mask | (
                        segmentations[:, index_segmentation] == concept_index
                    )
                masks[concept_index].append(concept_mask.cpu())
        for concept_index in sorted(masks.keys()):
            if concept_index < len(concept_names):
                # Prepare for scipy sparse matrix format
                masks[concept_index] = torch.cat(masks[concept_index], 0)
                masks[concept_index] = torch.reshape(
                    masks[concept_index], (masks[concept_index].shape[0], -1)
                )
                masks[concept_index] = masks[concept_index].numpy()
                with open(
                    f"{segmentations_directory}/"
                    + f"{concept_names[concept_index]}.npz",
                    "wb",
                ) as file:
                    sparse.save_npz(
                        file, sparse.csr_matrix(masks[concept_index])
                    )


def get_mask_type(masks):
    """
    Returns the type of the masks.
    Args:
        masks (list): list of masks.
    Returns:
        Type of the masks.
    """
    if isinstance(masks[1], sparse.csr_matrix):
        return "csr"
    else:
        return "torch"


def extract_mask(index, masks, mask_type="torch"):
    """
    Extracts the mask from the list of masks.
    Args:
        index (int): index of the mask to extract.
        masks (list): list of masks.
        mask_type (str): type of the masks.
    Returns:
        Mask.
    """
    if mask_type == "csr":
        return torch.from_numpy(masks[index].toarray())
    else:
        return masks[index]


def get_areas_mask(masks, info_directory, device=torch.device("cpu")):
    """
    Returns the areas per sample of the masks for each atomic concept.
    Args:
        masks (list): list of masks.
        info_directory (str): directory where the information is stored.
        device (torch.device): device to use.
    Returns:
        List of areas of the masks.
    """
    areas = [None]
    file_concept_areas = f"{info_directory}/concept_areas_list.pkl"
    if os.path.exists(file_concept_areas):
        with open(file_concept_areas, "rb") as file:
            areas = pickle.load(file)
    else:
        mask_type = get_mask_type(masks)
        for concept in range(1, len(masks)):
            areas.append(
                torch.sum(
                    extract_mask(concept, masks, mask_type),
                    1,
                    dtype=torch.int32,
                )
            )
        with open(file_concept_areas, "wb") as file:
            pickle.dump(areas, file)
    for i in range(len(areas)):
        if areas[i] is not None:
            areas[i] = areas[i].to(device)
    return areas


def get_overscribed_rectangles(masks, mask_shape):
    """
    Returns the vertices of the bounding box overscribed
    on the masks.

    Args:
        masks (torch.Tensor): tensor of masks.

    Returns:
        Tensor of vertices of the bounding box overscribed
        on the masks. The tensor has shape (num_samples, 2, 2).
        The first dimension corresponds to the sample index,
        the second and third dimension correspond to the top
        left and bottom right vertices of the bounding box,
        respectively.
    """
    num_samples = masks.shape[0]
    non_empty = torch.any(masks.reshape(num_samples, -1), 1)
    points = torch.tensor([[[0, 0], [0, 0]]] * num_samples)
    if non_empty.any():
        non_zero_points = torchvision.ops.masks_to_boxes(
            masks[non_empty].reshape(-1, mask_shape[0], mask_shape[1])
        )
        top_left = non_zero_points[:, 0:2]
        bottom_right = non_zero_points[:, 2:4]
        points[non_empty] = torch.tensor(
            list(zip(top_left.tolist(), bottom_right.tolist()))
        ).long()
    return points


def get_inscribed_rectangles(matrices, device):
    """ Returns the vertices of the bounding box inscribed
    on the masks.

    Args:
        matrices (torch.Tensor): tensor of masks.

    Returns:
        Tensor of vertices of the bounding box inscribed
        on the masks. The tensor has shape (num_samples, 2, 2).
        The first dimension corresponds to the sample index,
        the second and third dimension correspond to the top
        left and bottom right vertices of the bounding box,
        respectively.
    """
    torch_num_samples = matrices.shape[0]
    torch_num_columns = matrices.shape[2]
    torch_num_rows = matrices.shape[1]

    # initialize left as the leftmost boundary possible
    torch_left = torch.zeros(
        torch_num_samples, torch_num_columns, device=device)
    # initialize right as the rightmost boundary possible
    torch_right = torch.zeros(
        torch_num_samples,
        torch_num_columns,
        device=device) + torch_num_columns
    torch_height = torch.zeros(
        torch_num_samples,
        torch_num_columns,
        device=device)
    torch_col_indices = torch.arange(
        torch_num_columns, device=device).repeat(
            torch_num_samples, 1)
    torch_max_area = torch.zeros(torch_num_samples, device=device)

    bottom_right_row = torch.zeros(
        torch_num_samples, device=device).int() - 1
    bottom_right_col = torch.zeros(
        torch_num_samples, device=device).int() - 1
    top_left_row = torch.zeros(torch_num_samples, device=device).int()
    top_left_col = torch.zeros(torch_num_samples, device=device).int()

    torch_col_indices = torch.arange(
        torch_num_columns, device=device).repeat(
            torch_num_samples, torch_num_rows, 1)

    # get indices of the closer zero from the left
    zero_indices_left = torch.zeros(
        torch_num_samples, torch_num_rows, torch_num_columns, device=device)
    for j in range(1, torch_num_columns):
        zero_indices_left[:, :, j] = torch.where(
            matrices[:, :, j-1] == False,
            torch_col_indices[:, :, j],
            zero_indices_left[:, :, j-1])

    # get indices of the closer zero from the right
    zero_indices_right = torch.zeros(
        torch_num_samples,
        torch_num_rows,
        torch_num_columns,
        device=device) + torch_num_columns
    for j in range(torch_num_columns-2, -1, -1):
        zero_indices_right[:, :, j] = torch.where(
            matrices[:, :, j+1] == False,
            torch_col_indices[:, :, j+1],
            zero_indices_right[:, :, j+1])

    for i in range(torch_num_rows):
        current_row = matrices[:, i, :]
        # update height
        torch_height = torch.where(current_row == True, torch_height + 1, 0)

        # update left
        torch_left = torch.where(
            current_row == True,
            torch.maximum(torch_left, zero_indices_left[:, i, :]),
            0)

        # update right
        torch_right = torch.where(
            current_row == True,
            torch.minimum(
                torch_right,
                zero_indices_right[:, i, :]),
            torch_num_columns)

        # update the area
        for j in range(torch_num_columns):
            area = torch_height[:, j] * (torch_right[:, j] - torch_left[:, j])
            bottom_right_col = torch.where(
                area > torch_max_area,
                torch_right[:, j]-1,
                bottom_right_col)
            bottom_right_row = torch.where(
                area > torch_max_area,
                i,
                bottom_right_row)
            top_left_row = torch.where(
                area > torch_max_area,
                bottom_right_row - torch_height[:, j]+1,
                top_left_row)
            top_left_col = torch.where(
                area > torch_max_area,
                bottom_right_col - (torch_right[:, j] - torch_left[:, j])+1,
                top_left_col)
            torch_max_area = torch.maximum(torch_max_area, area)

    top_left = torch.tensor(
        list(
            zip(
                top_left_row.tolist(),
                top_left_col.tolist())
            ))
    bottom_right = torch.tensor(
        list(
            zip(
                bottom_right_row.tolist(),
                bottom_right_col.tolist()
                )
            ))
    return torch.tensor(list(zip(top_left.tolist(), bottom_right.tolist())))


def get_concept_inscribed_masks(masks, mask_shape, info_directory, device):
    inscribed = [None]
    file_path = f"{info_directory}/positive_inscripted.pkl"
    if os.path.exists(file_path):
        with open(file_path, "rb") as file:
            inscribed = pickle.load(file)
    else:
        mask_type = get_mask_type(masks)
        for concept in tqdm(
            range(1, len(masks)),
            total=len(masks) - 1,
            desc="Getting inscribed masks",
        ):
            concept_masks = extract_mask(concept, masks, mask_type)
            concept_masks = torch.reshape(
                concept_masks, (-1, mask_shape[0], mask_shape[1])
            ).to(device)
            inscribed.append(
                get_inscribed_rectangles(concept_masks, device).numpy()
            )
        with open(file_path, "wb") as file:
            pickle.dump(inscribed, file)
    for i in range(1, len(inscribed)):
        inscribed[i] = torch.from_numpy(inscribed[i]).to(device)
    return inscribed


def get_masks_info(masks, config):
    """ Returns the masks information useful for the heuristics.

    Args:
        masks (list): list of masks.
        config (src.config.Config): configuration of the current run.

    Returns:
        tuple: tuple containing:
            - concept_areas (list): list of areas of the masks.
            - inscribed_rectangles (list): list of inscribed
                rectangles of the masks.
            - bounding_boxes (list): list of bounding boxes of the masks.
    """
    mask_shape = config.get_mask_shape()
    directory = config.get_info_directory()
    device = config.device
    concept_areas = get_areas_mask(masks, directory, device)
    inscribed_rectangles = get_concept_inscribed_masks(
        masks, mask_shape=mask_shape, info_directory=directory, device=device
    )
    bounding_boxes = get_bounding_boxes(
        masks, mask_shape=mask_shape, info_directory=directory, device=device
    )
    masks_info = (concept_areas, (inscribed_rectangles, bounding_boxes))
    return masks_info


def get_bounding_boxes(masks, mask_shape, info_directory, device):
    """ Returns the bounding boxes of the masks.

    Args:
        masks (list): list of masks.
        mask_shape (tuple): shape of a mask.
        info_directory (str): directory where to save/load the information.
        device (torch.device): device to use.

    Returns:
        list: list of bounding boxes of the masks.
        """
    overscribed = [None]
    file_path = f"{info_directory}/positive_rectangles.pkl"
    if os.path.exists(file_path):
        with open(file_path, "rb") as file:
            overscribed = pickle.load(file)
    else:
        mask_type = get_mask_type(masks)
        for concept in tqdm(
            range(1, len(masks)),
            total=len(masks) - 1,
            desc="Getting bounding box for masks",
        ):
            concept_masks = extract_mask(concept, masks, mask_type)
            concept_masks = torch.reshape(
                concept_masks, (-1, mask_shape[0], mask_shape[1])
            )
            overscribed.append(
                get_overscribed_rectangles(concept_masks, mask_shape).numpy()
            )
        with open(file_path, "wb") as file:
            pickle.dump(overscribed, file)
    for i in range(1, len(overscribed)):
        overscribed[i] = torch.from_numpy(overscribed[i]).to(device)
    return overscribed


def get_formula_mask(f, masks, optional_masks=None):
    """
    Function to return a mask for a given formula.
    Args:
        f (src.formula.Formula): formula.
        masks (list): list of masks.
        optional_masks (dict): dictionary of additional masks (beam masks).
    Returns:
        Formula's Mask.
    """
    if optional_masks is not None and f in optional_masks.keys():
        mask = optional_masks[f]
        if isinstance(mask, sparse.csr.csr_matrix):
            return utils.sparse_to_torch(mask)
        else:
            return mask
    if isinstance(f, F.Leaf):
        mask = masks[f.val]
        if isinstance(mask, sparse.csr.csr_matrix):
            return utils.sparse_to_torch(mask)
        else:
            return mask
    elif isinstance(f, F.Or):
        masks_l = get_formula_mask(f.left, masks, optional_masks)
        masks_r = get_formula_mask(f.right, masks, optional_masks)
        return masks_l | masks_r
    elif isinstance(f, F.And):
        masks_l = get_formula_mask(f.left, masks, optional_masks)
        masks_r = get_formula_mask(f.right, masks, optional_masks)
        return masks_l & masks_r
    elif isinstance(f, F.Not):
        return ~get_formula_mask(f.val, masks, optional_masks)
    elif isinstance(f, int):
        mask = masks[f]
        if isinstance(mask, sparse.csr.csr_matrix):
            return utils.sparse_to_torch(mask)
        else:
            return mask
    else:
        raise ValueError(f"Unknown formula type {type(f)}")


def get_masks(masks_directory, dataloader, labels, device, pre_load=False):
    """
    Returns the masks for the given dataloader and labels.
    Args:
        masks_directory (str): directory where the sparse masks are stored.
        dataloader (torch.utils.data.DataLoader): dataloader for the images.
        labels (list): list of labels.
    Returns:
        List of masks.
    """
    if not os.path.exists(masks_directory):
        os.makedirs(masks_directory)
    # If some file is missing, generate again the sparse masks
    if len(os.listdir(masks_directory)) != len(labels):
        print("Generating and saving sparse masks")
        save_sparse_masks(dataloader, labels, masks_directory, device)
    masks = load_sparse_masks(labels, masks_directory)

    # Pre-load the masks in memory. RAM-intensive but slightly faster
    if pre_load:
        for i in range(1, len(masks)):
            masks[i] = torch.from_numpy(masks[i].toarray())
    return masks
