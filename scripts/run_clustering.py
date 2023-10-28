"""Script to run the clustering algorithm for compositional explanations.
"""

import os
import pickle
import random

import torch
import torchvision
import absl.flags
import absl.app
from tqdm import tqdm

from src import segmentations
from src import model_utils
from src import mask_utils
from src import activation_utils
from src import algorithms
from src import utils
from src import formula as F
from src import settings
from src import constants as C

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

absl.flags.DEFINE_integer("length", 3, "length of explanations")
absl.flags.DEFINE_integer("num_clusters", 5, "number of clusters")
absl.flags.DEFINE_integer("beam_limit", 5, "beam limit")
absl.flags.DEFINE_integer("random_units", 0, "number of units")
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

FLAGS = absl.flags.FLAGS


def main(argv):
    if FLAGS.num_clusters < 1:
        raise ValueError("num_clusters must be greater than 0")
    # Set seed
    generator = utils.set_seed(FLAGS.seed)

    # Parameters
    cfg = settings.Settings(
        subset=FLAGS.subset,
        model=FLAGS.model,
        pretrained=FLAGS.pretrained,
        num_clusters=FLAGS.num_clusters,
        beam_limit=FLAGS.beam_limit,
        device=FLAGS.device,
        root_models=FLAGS.root_models,
        root_datasets=FLAGS.root_datasets,
        root_segmentations=FLAGS.root_segmentations,
        root_activations=FLAGS.root_activations,
        root_results=FLAGS.root_results
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
        batch_size=C.BATCH_SIZE,
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
    masks_info = mask_utils.get_masks_info(masks, config=cfg)

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
                range(num_units), FLAGS.random_units
            )

        for unit in tqdm(
            selected_units, desc="Computing Compostional explanations per unit"
        ):
            unit_activations = activations[unit]

            # Compute activation range to be kept in the masks
            activation_ranges = activation_utils.compute_activation_ranges(
                unit_activations, FLAGS.num_clusters)

            # Loop over all the activation ranges
            for cluster_index, activation_range in enumerate(
                sorted(activation_ranges)
            ):
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
                        heuristic="mmesh",
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
                string_label = F.get_formula_str(best_label, dataset.labels)
                print(
                    f"Parsed Unit: {unit} - "
                    + f"Cluster: {cluster_index} - "
                    + f"Best Label: {string_label} - "
                    + f"Best IoU: {round(best_iou,3)} - "
                    + f"Visited: {visited}"
                )


if __name__ == "__main__":
    with torch.no_grad():
        absl.app.run(main)
