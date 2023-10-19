""" Script to run the threshold comparison experiment.
The scripts compares composotional explanation for the following
list of quantiles:
[0.005, 0.01, 0.05, 0.1, 0.2, 0.5, 0.8, 0.9, 0.95, 0.99, 0.995]
"""

from collections import Counter, defaultdict
import os
import pickle
import random

import torch
import torch.multiprocessing as mp
import torchvision
import absl.flags
import absl.app
from tqdm import tqdm
import matplotlib.pyplot as plt

from src import segmentations
from src import model_utils
from src import mask_utils
from src import activation_utils
from src import utils
from src import formula as F
from src import settings
from src import algorithms


# user flags
absl.flags.DEFINE_string(
    "subset", "ade20k", "subset to use. Values:[ade20k, pascal]"
)
absl.flags.DEFINE_string(
    "model",
    "resnet18",
    "model to use. Values:[resnet18, alexnet, resnet50, vgg16, densenet161]",
)
absl.flags.DEFINE_string(
    "pretrained",
    "places365",
    "whether to use pretrained weights. Values [imagenet, places365, None]",
)
absl.flags.DEFINE_string("device", "cuda", "device to use to store the model")
absl.flags.DEFINE_string(
    "heuristic", "mmesh", "heuristic to use. Values:[mmesh, None]"
)
absl.flags.DEFINE_integer("length", 3, "length of explanations")
absl.flags.DEFINE_integer("random_units", 50, "number of units")
absl.flags.DEFINE_string(
    "root_models", "data/model/", "root directory for models"
)
absl.flags.DEFINE_string(
    "root_datasets", "data/dataset/", "root directory for datasets"
)
absl.flags.DEFINE_string(
    "root_segmentations",
    "data/cache/segmentations/",
    "root directory for segmentations",
)
absl.flags.DEFINE_string(
    "root_activations",
    "data/cache/activations/",
    "root directory for activations",
)
absl.flags.DEFINE_string(
    "root_results", "data/results/", "root directory for results"
)
absl.flags.DEFINE_integer("seed", 0, "seed to use to set reproducibility")
absl.flags.DEFINE_string(
    "filename_figure", "grid_threshold.png", "filename to save the plot"
)

FLAGS = absl.flags.FLAGS
QUANTILES = [0.005, 0.01, 0.05, 0.1, 0.2, 0.5, 0.8, 0.9, 0.95, 0.99, 0.995]


def grid_threshold_plot(quantiles, threshold_categories, path):
    """
    Plots the percentage of labels that are above the threshold
    for each category.

    Args:
        quantiles (list): list of quantiles used to threshold
        threshold_categories (dict): dictionary of categories and
            their thresholded labels
        path (str): path to save the plot
    """
    object_up = []
    color_up = []
    scene_up = []
    part_up = []
    material_up = []
    for quantile in quantiles[:6]:
        counter = Counter(threshold_categories[quantile])
        covered = (
            counter["object"]
            + counter["color"]
            + counter["scene"]
            + counter["part"]
            + counter["material"]
        )
        object_up.append(counter["object"] / covered)
        color_up.append(counter["color"] / covered)
        scene_up.append(counter["scene"] / covered)
        part_up.append(counter["part"] / covered)
        material_up.append(counter["material"] / covered)
    object_down = []
    color_down = []
    scene_down = []
    part_down = []
    material_down = []
    low_quantiles = quantiles[5:]
    for quantile in low_quantiles[::-1]:
        counter = Counter(threshold_categories[quantile])
        covered = (
            counter["object"]
            + counter["color"]
            + counter["scene"]
            + counter["part"]
            + counter["material"]
        )
        object_down.append(counter["object"] / covered)
        color_down.append(counter["color"] / covered)
        scene_down.append(counter["scene"] / covered)
        part_down.append(counter["part"] / covered)
        material_down.append(counter["material"] / covered)
    fig, axs = plt.subplots(1, 2)
    fig.set_size_inches(8, 4)
    axs[0].plot(quantiles[:6], object_up, label="object")
    axs[0].plot(quantiles[:6], color_up, label="color")
    axs[0].plot(quantiles[:6], scene_up, label="scene")
    axs[0].plot(quantiles[:6], part_up, label="part")
    axs[0].plot(quantiles[:6], material_up, label="material")
    axs[0].set_ylim(0, 1)
    axs[0].set_xlabel("High Quantile")
    axs[0].set_ylabel("% of labels")
    axs[1].plot(quantiles[:6], object_down, label="object")
    axs[1].plot(quantiles[:6], color_down, label="color")
    axs[1].plot(quantiles[:6], scene_down, label="scene")
    axs[1].plot(quantiles[:6], part_down, label="part")
    axs[1].plot(quantiles[:6], material_down, label="material")
    axs[1].set_ylim(0, 1)
    axs[1].set_xlabel("Low Quantile")

    # Set horizontal legend at the bottom of the figure.
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=len(labels), loc="upper center")

    fig.savefig(path)
    plt.close()


def get_positive_vals(f):
    """
    Function to return a mask for a given formula.
    """
    if isinstance(f, F.Leaf):
        return [f.val]
    elif isinstance(f, F.Not):
        return []
    elif isinstance(f, F.BinaryNode):
        vals = []
        val_l = get_positive_vals(f.left)
        val_r = get_positive_vals(f.right)
        vals.extend(val_l)
        vals.extend(val_r)
        return vals


def main(argv):
    # Set seed
    generator = utils.set_seed(FLAGS.seed)

    # Parameters
    cfg = settings.Settings(
        subset=FLAGS.subset,
        model=FLAGS.model,
        pretrained=FLAGS.pretrained,
        num_clusters=0,
        beam_limit=5,
        device=FLAGS.device,
        root_models=FLAGS.root_models,
        root_datasets=FLAGS.root_datasets,
        root_segmentations=FLAGS.root_segmentations,
        root_activations=FLAGS.root_activations,
        root_results=FLAGS.root_results,
    )
    sparse_segmentation_directory = cfg.get_segmentation_directory()
    mask_shape = cfg.get_mask_shape()

    # Load data
    dataset = segmentations.BrodenDataset(
        cfg.dir_datasets,
        subset=cfg.index_subset,
        resolution=cfg.get_img_size(),
        broden_version=1,
        transform_image=torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize(cfg.get_img_size()),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(
                    cfg.get_image_mean(), cfg.get_image_stdev()
                ),
            ]
        ),
    )
    segmentation_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=32,
        worker_init_fn=utils.seed_worker,
        generator=generator,
    )

    # Load Model
    model = model_utils.load_model_from_settings(cfg, device=cfg.device)

    # Load Masks
    masks = mask_utils.get_masks(
        sparse_segmentation_directory, segmentation_loader, dataset.labels
    )

    # Get Masks Information from the concept dataset
    if FLAGS.length > 1:
        concept_areas, (
            inscribed_rectangles,
            bounding_boxes,
        ) = mask_utils.get_masks_info(masks, config=cfg)
    else:
        concept_areas, (inscribed_rectangles, bounding_boxes) = None, (
            None,
            None,
        )
    if FLAGS.heuristic == "mmesh":
        masks_info = (concept_areas, (inscribed_rectangles, bounding_boxes))
    elif FLAGS.heuristic == "cfh" or FLAGS.heuristic == "areas":
        masks_info = concept_areas
    else:
        masks_info = None

    # Loop over all the selected layers
    for _, layer_name in enumerate(cfg.get_feature_names()):
        # Get the number of units in the layer
        num_units = model_utils.get_number_of_units(model, layer_name, cfg)

        # Get activations
        activations = model_utils.get_layer_activations(
            segmentation_loader,
            model,
            layer_name,
            range(num_units),
            cfg.get_activation_directory(),
        )

        # Select units
        if FLAGS.random_units == 0:
            selected_units = range(num_units)
        else:
            selected_units = random.sample(
                range(0, num_units), FLAGS.random_units
            )

        threshold_categories = defaultdict(list)
        for unit in tqdm(
            selected_units,
            desc="Computing Compostional explanations per unit",
        ):
            # Get unit activations
            unit_activations = activations[unit]

            # Compute thresholds
            thresholds = [
                activation_utils.quantile_threshold(
                    unit_activations, quantile, avoid_zero=True
                )[0]
                for quantile in QUANTILES
            ]
            assigned_categories = defaultdict(list)
            for threshold, quantile in zip(thresholds, QUANTILES):
                activation_range = (
                    (threshold, torch.tensor(float("inf")))
                    if quantile <= 0.5
                    else (1e-6, threshold)
                )
                dir_current_results = (
                    f"{cfg.get_results_directory()}/"
                    + f"{layer_name}/{unit}/{activation_range}"
                )
                if not os.path.exists(dir_current_results):
                    os.makedirs(dir_current_results)
                file_algo_results = (
                    f"{dir_current_results}/" + f"{FLAGS.length}.pickle"
                )
                if not os.path.exists(file_algo_results):
                    # Compute binary masks
                    bitmaps = activation_utils.compute_bitmaps(
                        unit_activations,
                        activation_range,
                        mask_shape=mask_shape,
                    )
                    bitmaps = bitmaps.to(cfg.device)
                    (
                        best_label,
                        best_iou,
                        visited,
                    ) = algorithms.get_heuristic_scores(
                        masks,
                        bitmaps,
                        segmentations_info=masks_info,
                        heuristic=FLAGS.heuristic,
                        length=FLAGS.length,
                        max_size_mask=cfg.get_max_mask_size(),
                        mask_shape=cfg.get_mask_shape(),
                        device=cfg.device,
                    )
                    with open(file_algo_results, "wb") as file:
                        pickle.dump((best_label, best_iou, visited), file)
                else:
                    with open(file_algo_results, "rb") as file:
                        best_label, best_iou, visited = pickle.load(file)
                if isinstance(best_label, int):
                    assigned_categories[quantile].append(best_label)
                else:
                    for label in get_positive_vals(best_label):
                        assigned_categories[quantile].append(label)
            for quantile in QUANTILES:
                for best_label in assigned_categories[quantile]:
                    label_category = dataset.label_category[best_label]
                    label_category = dataset.categories[label_category]
                    threshold_categories[quantile].append(label_category)
                print(
                    f"Quantile:{quantile} " +
                    f"{Counter(threshold_categories[quantile])}"
                )

        # generate figure
        grid_threshold_plot(
            QUANTILES,
            threshold_categories,
            path=f"figures/{FLAGS.filename_figure}",
        )


if __name__ == "__main__":
    mp.set_start_method("spawn")
    absl.app.run(main)
