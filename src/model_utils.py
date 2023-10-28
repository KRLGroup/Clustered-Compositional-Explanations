""" This module contains the functions to load the model and to compute the
activations of the model.
"""
import os
from typing import List
import requests

import torch
import torchvision
import numpy as np
import skimage

from src import settings


# Reference: https://github.com/jayelm/compexp/blob/master/vision/loader/model_loader.py
def load_model_from_settings(
    config,
    device=torch.device("cuda"),
):
    """
    Load the model from the settings.
    Args:
        config (settings.Settings): current config of the settings
        device (torch.device): device to use

    Returns:
        torch.nn.Module: model
    """
    model_name = config.model
    weights = config.get_weights()
    parallel = config.get_parallel()
    num_classes = config.get_num_classes()
    model_file_path = config.get_model_file_path()
    pretrained = config.pretrained
    
    if pretrained == "places365" and not os.path.exists(model_file_path):
        if not os.path.exists(config.get_model_root()):
            os.makedirs(config.get_model_root())
        raise FileNotFoundError(f"Model file not found: {model_file_path}")
    print(pretrained)
    if pretrained == "places365":
        print(f"Loading model:{model_name}\n\tfrom {model_file_path}")
    elif pretrained == "imagenet":
        print(
            f"Loading model:{model_name}\n\tfrom imagenet pre-trained weights"
        )
    else:
        print(f"Loading UNTRAINED model:{model_name}\n")

    model_fn = torchvision.models.__dict__[model_name]

    if weights == "IMAGENET1K_V1":
        model = model_fn(weights=weights)
    elif weights is None:
        model = model_fn(pretrained=False, num_classes=num_classes)
    else:
        checkpoint = torch.load(weights, map_location=device)
        if (
            type(checkpoint).__name__ == "OrderedDict"
            or type(checkpoint).__name__ == "dict"
        ):
            model = model_fn(num_classes=num_classes)
            if parallel:
                # the data parallel layer will add 'module' before each
                # layer name
                state_dict = {
                    str.replace(k, "module.", ""): v
                    for k, v in checkpoint["state_dict"].items()
                }
            else:
                state_dict = checkpoint
            model.load_state_dict(state_dict)
        else:
            if model_name == "densenet161":
                # Fix old densenet pytorch names.
                model = model_fn(num_classes=num_classes)
                state_dict = checkpoint.state_dict()

                def rep(k):
                    for i in range(6):
                        k = k.replace(f"norm.{i}", f"norm{i}")
                        k = k.replace(f"relu.{i}", f"relu{i}")
                        k = k.replace(f"conv.{i}", f"conv{i}")
                    return k

                state_dict = {rep(k): v for k, v in state_dict.items()}
                model.load_state_dict(state_dict)
            else:
                model = checkpoint
    model = model.to(device)
    model.eval()
    print(f"{model_name} loaded. Eval Modality")
    return model


# Reference: https://github.com/jayelm/compexp/blob/master/vision/loader/model_loader.py
def hook(model, hook_fn, feature_names):
    """
    Register a hook to a model.

    Args:
        model (torch.nn.Module): model
        hook_fn (function): hook function
        feature_names (list): list of feature names

    Returns:
        list: list of handles
    """
    handles = []
    for name in feature_names:
        if isinstance(name, list):
            # Iteratively retrive the module
            hook_model = model
            for n in name:
                hook_model = hook_model._modules.get(n)
        else:
            hook_model = model._modules.get(name)
        if hook_model is None:
            raise ValueError(f"Couldn't find feature {name}")
        handles.append(hook_model.register_forward_hook(hook_fn))
    return handles


@torch.no_grad()
def get_number_of_units(
        model: torch.nn.Module, layer_name: str,
        config: settings.Settings):
    """Get the number of units of a given layer.

    Args:
        model (torch.nn.Module): model
        layer_name (str): the layer name
        config (settings.Settings): settings use to retrieve the image size

    Returns:
        int: number of units
    """
    dummy = torch.randn(1, 3, config.get_img_size(), config.get_img_size()).to(
        config.device
    )
    activations = []

    def hook_feature(module, inp, output):
        activations.append(output.data.cpu())

    handles = hook(model, hook_feature, [layer_name])
    model(dummy)
    num_units = activations[0].shape[1]
    for handle in handles:
        handle.remove()
    return num_units


@torch.no_grad()
def compute_activations(
        loader: torch.utils.data.DataLoader, model: torch.nn.Module,
        layers: List, units: List[int] = None):
    """Retrieve the activations of a given layer feeding the model with the
    images in the loader.

    Args:
        loader (torch.utils.data.DataLoader): data loader
        model (torch.nn.Module): model from which the activations are computed
        layers (list): list of layers
        units (list): list of units whose activations are to be computed

    Returns:
        torch.Tensor: activations
    """
    device = next(model.parameters()).device
    temp_activations = []

    def hook_feature(module, inp, output):
        if units is not None:
            temp_activations.append(output[:, units].data.cpu())
        else:
            temp_activations.append(output.data.cpu())

    handles = hook(model, hook_feature, layers)

    activations = [[] for _ in range(len(layers))]
    for _, (images, _, concept_matrix) in enumerate(loader):
        # Transformations
        images = images[concept_matrix.sum(1) > 0]

        # Move to GPU
        images = images.to(device)

        # Forward pass
        _ = model(images)

        # collect data
        for index_layer in range(len(layers)):
            activations[index_layer].append(temp_activations[index_layer])

        # Empty the temp list
        del temp_activations[:]
        temp_activations = []

    for handle in handles:
        handle.remove()

    for layer in range(len(layers)):
        activations[layer] = torch.unsqueeze(
            torch.cat(activations[layer]), dim=0
        )
    activations = torch.cat(activations, dim=0)
    return activations


def get_layer_activations(loader, model, layer, units, dir_activations):
    """Checks if the activations are already computed and saved, otherwise
    computes them and saves them.

    Args:
        loader (torch.utils.data.DataLoader): data loader
        model (torch.nn.Module): model
        layer (str): layer name
        units (list): list of units whose activations are to be computed

    Returns:
        activations (list): list of activations for each unit
    """
    layer_dir = f"{dir_activations}/{layer}"
    units_to_compute = []
    saved_units = []
    total_activations = [[None] for _ in range(max(units) + 1)]
    for unit in units:
        if not os.path.exists(f"{layer_dir}/{unit}.npy"):
            print(f"{layer_dir}/{unit}.npy")
            units_to_compute.append(unit)
        else:
            saved_units.append(unit)
            total_activations[unit] = torch.from_numpy(
                np.load(f"{layer_dir}/{unit}.npy")
            )
    if len(units_to_compute) > 0:
        print(f"Computing activations for units {units_to_compute}")

        activations = compute_activations(
            loader, model, [layer], units=units_to_compute
        )
        # since the function checks one layer at a time
        activations = activations[0]
        if not os.path.exists(layer_dir):
            os.makedirs(layer_dir)

        for index, unit in enumerate(units_to_compute):
            np.save(f"{layer_dir}/{unit}.npy", activations[:, index])
            total_activations[unit] = activations[:, index]
    total_activations = torch.stack(total_activations, 0)

    return total_activations


@torch.no_grad()
def apply_concept_masking(
    loader, model, layers, units=None, mask=None, image_size=224
):
    """ Retrieve the activations of a given layer feeding the model with the
    images in the loader and applying a mask to the images.

    Args:
        loader (torch.utils.data.DataLoader): data loader
        model (torch.nn.Module): model from which the activations are computed
        layers (list): list of layers
        units (list): list of units whose activations are to be computed
        mask (torch.Tensor): mask to apply to the images
        image_size (int): image size

    Returns:
        torch.Tensor: activations
    """
    device = next(model.parameters()).device
    temp_activations = []

    def hook_feature(module, inp, output):
        if units is not None:
            temp_activations.append(output[:, units].data.cpu())
        else:
            temp_activations.append(output.data.cpu())
        del output

    handles = hook(model, hook_feature, layers)
    activations = [[] for _ in range(len(layers))]
    if mask.dtype == torch.bool:
        upsampled_masks = skimage.transform.resize(
            mask.cpu().numpy(),
            (mask.shape[0], image_size, image_size),
            order=0,
            preserve_range=True,
        )
    else:
        upsampled_masks = skimage.transform.resize(
            mask.cpu().numpy(),
            (mask.shape[0], image_size, image_size),
            order=1,
        )
    upsampled_masks = torch.from_numpy(upsampled_masks).float()
    upsampled_masks = upsampled_masks.unsqueeze(1)
    upsampled_masks = upsampled_masks.repeat(1, 3, 1, 1)
    for batch_idx, (images, _, _) in enumerate(loader):
        if mask is not None:
            formula_mask = upsampled_masks[
                batch_idx
                * loader.batch_size: (batch_idx + 1)
                * loader.batch_size
            ]
            images = images * formula_mask

        # Move to GPU
        images = images.to(device)

        # Forward pass
        outputs = model(images)

        # collect data
        for index_layer in range(len(layers)):
            activations[index_layer].append(temp_activations[index_layer])

        # Empty the temp list
        del temp_activations[:]
        del outputs
        temp_activations = []

    for handle in handles:
        handle.remove()

    for layer in range(len(layers)):
        activations[layer] = torch.unsqueeze(
            torch.cat(activations[layer]), dim=0
        )
    activations = torch.cat(activations, dim=0)
    activations = activations.to(device)
    return activations
