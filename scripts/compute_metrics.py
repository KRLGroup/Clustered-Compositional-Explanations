""" Compute metrics for explanations generated by
 run_compositional_explanations.py"""
import os
import pickle
import csv
from collections import defaultdict
import random

import numpy as np
import torch
import torchvision
import absl.flags
import absl.app
from tqdm import tqdm

from src import segmentations
from src import model_utils
from src import mask_utils
from src import activation_utils
from src import metrics
from src import utils
from src import formula as F
from src import settings


# User flags
absl.flags.DEFINE_string(
    "dataset", "places365", "dataset to use. Values:[places365, imagenet]"
)
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


def print_summary(total_scores, clusters_scores):
    """Print summary of the metrics computed so far

    Args:
        total_scores (dict): dictionary containing the total scores
        clusters_scores (dict): dictionary containing the scores per cluster

    Returns:
        None
    """
    def camelcase(word):
        list_words = word.split("_")
        converted = " ".join(
            word[0].upper() + word[1:].lower() for word in list_words
        )
        return converted

    print("******************************")
    print("Total scores")
    for metric, scores in total_scores.items():
        print(
            f"{camelcase(metric)}: "
            + f"Avg: {round(np.mean(scores), 2)} "
            + f"StdDev: {round(np.std(scores), 2)}"
        )

    print()
    print("Clusters scores")
    for cluster, scores_dict in clusters_scores.items():
        print(f"\nCluster {cluster}")
        for metric, scores in scores_dict.items():
            print(
                f"{camelcase(metric)}: "
                + f"Avg: {round(np.mean(scores), 2)} "
                + f"StdDev: {round(np.std(scores), 2)}"
            )
    print("******************************")


def main(argv):
    # Set seed
    utils.set_seed(FLAGS.seed)

    # Parameters
    device = FLAGS.device

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
        root_results=FLAGS.root_results,
    )
    sparse_segmentation_directory = cfg.get_segmentation_directory()
    mask_shape = cfg.get_mask_shape()
    image_size = cfg.get_img_size()

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
    segmentation_loader = torch.utils.data.DataLoader(dataset, batch_size=32)
    # Load Model
    model = model_utils.load_model_from_settings(cfg, device=device)

    sparse_segmentation_directory = cfg.get_segmentation_directory()
    mask_shape = cfg.get_mask_shape()

    if not os.path.exists(sparse_segmentation_directory):
        os.makedirs(sparse_segmentation_directory)

    # If some file is missing, generate again the sparse masks
    masks = mask_utils.get_masks(
        sparse_segmentation_directory, segmentation_loader, dataset.labels,
        cfg.device
    )

    # Loop over all the selected layers
    for _, layer_name in enumerate(cfg.get_feature_names()):
        # Get the number of units in the layer
        num_units = model_utils.get_number_of_units(model, layer_name, cfg)

        # Select units
        if FLAGS.random_units == 0:
            selected_units = list(range(num_units))
        else:
            selected_units = random.sample(
                range(0, num_units), FLAGS.random_units
            )

        activations = model_utils.get_layer_activations(
            segmentation_loader,
            model,
            layer_name,
            range(num_units),
            cfg.get_activation_directory(),
        )

        csv_file = (
            f"{cfg.get_results_directory()}/"
            + f"{layer_name}_{FLAGS.num_clusters}_{FLAGS.length}"
            + "_metrics.csv"
        )

        total_scores = defaultdict(list)
        clusters_scores = defaultdict(lambda: defaultdict(list))

        # Load previous results if any
        file_already_exists = os.path.exists(csv_file)
        done_unit = {}
        if file_already_exists:
            print("Loading previous results")
            with open(csv_file, "r") as data:
                csv_reader = csv.DictReader(data)
                for line in csv_reader:
                    u = int(line.pop("unit"))
                    if u not in done_unit:
                        done_unit[u] = []
                    if u not in selected_units:
                        continue
                    cluster = int(line.pop("num_cluster"))
                    done_unit[u].append(cluster)
                    for column, score in line.items():
                        total_scores[column].append(float(score))
                        clusters_scores[cluster][column].append(float(score))
            # Print Loaded results
            print("Loaded results:")
            print_summary(total_scores, clusters_scores)

        # Remove units already done
        for unit in done_unit:
            if len(done_unit[unit]) == FLAGS.num_clusters:
                selected_units.remove(unit)
        print(f"Units to parse: {selected_units}")

        # Compute metrics for the remaining units
        for unit in tqdm(
            selected_units, desc="Computing Compostional explanations per unit"
        ):
            # Get activation ranges from clusters
            unit_activations = activations[unit]
            activation_ranges = activation_utils.compute_activation_ranges(
                unit_activations, FLAGS.num_clusters)
            activation_ranges = sorted(activation_ranges)
            total_samples = torch.zeros(len(dataset)).bool().to(device)
            for cluster_index, activation_range in enumerate(
                activation_ranges
            ):
                if unit in done_unit and cluster_index in done_unit[unit]:
                    print(unit, cluster_index, "already done")
                    continue
                print(f"parsing {unit} {cluster_index}")
                # Loop over all the activation ranges
                dir_current_results = (
                    f"{cfg.get_results_directory()}"
                    + f"/{layer_name}"
                    + f"/{unit}/"
                    + f"{activation_range}"
                )
                if not os.path.exists(dir_current_results):
                    os.makedirs(dir_current_results)
                file_algo_results = (
                    f"{dir_current_results}/{FLAGS.length}.pickle"
                )

                # Compute binary masks
                bitmaps = activation_utils.compute_bitmaps(
                    unit_activations, activation_range, mask_shape=mask_shape
                )
                bitmaps = bitmaps.to(device)
                unit_activations = unit_activations.to(device)
                try:
                    with open(file_algo_results, "rb") as file:
                        best_label, best_iou, visited = pickle.load(file)
                except FileNotFoundError as exc:
                    raise FileNotFoundError(
                        f"File {file_algo_results} not found. Please run "
                        + "run_compositional_explanations.py first using "
                        + "the same parameters."
                    ) from exc
                str_label = F.get_formula_str(best_label, dataset.labels)
                print(
                    f"Unit: {unit} - "
                    + f"Cluster: {cluster_index} - "
                    + f"Best Label: {str_label} - "
                    + f"Best IoU: {round(best_iou,3)} - "
                    + f"Visited: {visited}"
                )
                label_mask = mask_utils.get_formula_mask(best_label, masks).to(
                    device
                )
                iou = metrics.iou(bitmaps, label_mask)
                activation_coverage = metrics.activations_coverage(
                    bitmaps, label_mask
                )
                detection_accuracy = metrics.detection_accuracy(
                    bitmaps, label_mask
                )
                samples_coverage = metrics.samples_coverage(
                    bitmaps, label_mask
                )

                avg_segmentation_size = metrics.avg_mask_size(label_mask)
                avg_activation_size = metrics.avg_mask_size(bitmaps)
                avg_overlapping_activation_size = metrics.avg_mask_size(
                    bitmaps[(bitmaps & label_mask).sum(1) > 0]
                )
                avg_overlapping = metrics.avg_mask_size(bitmaps & label_mask)
                explanation_coverage = metrics.explanation_coverage(
                    bitmaps, label_mask)
                total_samples = (
                    total_samples | (bitmaps & label_mask).sum(1) > 0
                )
                dict_results = {
                    "unit": unit,
                    "iou": iou.item(),
                    "activation_coverage": activation_coverage.item(),
                    "label_coverage": detection_accuracy.item(),
                    "samples_coverage": samples_coverage.item(),
                    "avg_segmentation_size": avg_segmentation_size.item(),
                    "avg_activation_size": avg_activation_size.item(),
                    "avg_overlapping_activation_size":
                        avg_overlapping_activation_size.item(),
                    "avg_overlapping": avg_overlapping.item(),
                    "explanation_coverage": explanation_coverage.item(),
                    "num_cluster": cluster_index,
                }

                # concept activation
                masking_score = metrics.get_concept_masking(
                    activations=unit_activations,
                    mask_shape=mask_shape,
                    label_mask=label_mask,
                    loader=segmentation_loader,
                    model=model,
                    layer_name=layer_name,
                    unit=unit,
                    input_size=image_size,
                    activation_range=activation_range,
                )
                dict_results["label_masking"] = masking_score.item()
                for metric, score in dict_results.items():
                    total_scores[metric].append(score)
                for metric, score in dict_results.items():
                    clusters_scores[cluster_index][metric].append(score)

                # Print average so far
                print_summary(total_scores, clusters_scores)
                print("--------------------------------------------------")

                # Save results
                with open(csv_file, "a+") as f_object:
                    dictwriter = csv.DictWriter(
                        f_object, fieldnames=dict_results.keys()
                    )
                    if (
                        not file_already_exists
                    ):  # if file doesn't exist yet, write a header
                        dictwriter.writeheader()
                        file_already_exists = True
                    dictwriter.writerow(dict_results)
                    f_object.close()


if __name__ == "__main__":
    with torch.no_grad():
        absl.app.run(main)
