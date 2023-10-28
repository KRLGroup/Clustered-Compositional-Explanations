""" Script to compare the different heuristics """
import os
import time
import pickle
import random
from tqdm import tqdm
import torch
import torchvision
import numpy as np
import absl.flags
import absl.app

from src import segmentations
from src import settings
from src import model_utils
from src import mask_utils
from src import activation_utils
from src import algorithms
from src import utils

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
    "heuristic", "mmesh", "heuristic to use. Values:[mmesh, cfh, areas, None]"
)
absl.flags.DEFINE_integer("length", 3, "length of explanations")
absl.flags.DEFINE_integer("num_clusters", 1, "number of clusters")
absl.flags.DEFINE_integer("random_units", 100, "number of units")
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

        # Select 20 random units
        heuristic_avg_time = []
        heuristic_avg_visited = []
        with tqdm(
                selected_units,
                desc="Computing Compostional explanations per unit") as pbar:
            for unit in pbar:
                # Get unit activations
                unit_activations = activations[unit]
                activation_ranges = activation_utils.compute_activation_ranges(
                    unit_activations, FLAGS.num_clusters)

                random_activation_range = random.choice(activation_ranges)
                cluster_index = activation_ranges.index(
                    random_activation_range)

                dir_current_results = (
                    f"{cfg.get_results_directory()}/"
                    + f"{layer_name}/"
                    + f"{unit}/"
                    + f"{random_activation_range}"
                )
                if not os.path.exists(dir_current_results):
                    os.makedirs(dir_current_results)

                # The file path is different since the save info are different
                file_path = f"{dir_current_results}/" + \
                            f"h_{FLAGS.heuristic}_{FLAGS.length}.pickle"
                if not os.path.exists(file_path):
                    # Compute binary masks
                    bitmaps = activation_utils.compute_bitmaps(
                        unit_activations,
                        random_activation_range,
                        mask_shape=mask_shape,
                    )
                    bitmaps = bitmaps.to(cfg.device)
                    inner_init_time = time.time()
                    label, _, visited = algorithms.get_heuristic_scores(
                        masks,
                        bitmaps,
                        segmentations_info=masks_info,
                        heuristic=FLAGS.heuristic,
                        length=FLAGS.length,
                        max_size_mask=mask_shape[0] * mask_shape[1],
                        mask_shape=mask_shape,
                        device=cfg.device,
                    )
                    heuristic_time = (time.time() - inner_init_time) / 60
                    heuristic_result = (
                        (label, visited),
                        heuristic_time,
                        cluster_index,
                    )
                    heuristic_avg_time.append(heuristic_time)
                    heuristic_avg_visited.append(visited)

                    with open(file_path, "wb") as file:
                        pickle.dump(heuristic_result, file)
                else:
                    with open(file_path, "rb") as file:
                        (
                            (label, visited),
                            heuristic_time,
                            cluster_index,
                        ) = pickle.load(file)
                    heuristic_avg_visited.append(visited)
                    heuristic_avg_time.append(heuristic_time)
                pbar.update(0)
                pbar.set_postfix(
                    {
                        "Heuristic": FLAGS.heuristic,
                        "Avg time": np.mean(heuristic_avg_time),
                        "Avg visited": np.mean(heuristic_avg_visited),
                    }
                )
            print(
                f"Final results for layer {layer_name} "
                f"Heuristic {FLAGS.heuristic} "
                + f"Avg time: {np.mean(heuristic_avg_time):.2f} "
                + f"Std Dev: {np.std(heuristic_avg_time):.2f} "
                + f"Avg visited: {np.mean(heuristic_avg_visited):.2f} "
                + f"Std Dev: {np.std(heuristic_avg_visited):.2f} "
            )


if __name__ == "__main__":
    absl.app.run(main)
