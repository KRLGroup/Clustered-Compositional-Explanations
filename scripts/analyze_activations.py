"""Script to analyze the activations of a model.
The script computes the explanations for the untrained model and analyzes the
ones of a trained model in terms of default formulas.
"""

from collections import Counter
import pickle
import random
import math

import torch
import torchvision
import absl.flags
import absl.app
from tqdm import tqdm

from src import segmentations
from src import model_utils
from src import activation_utils
from src import utils
from src import formula as F
from src import settings


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
absl.flags.DEFINE_integer("num_clusters", 5, "number of clusters")
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
        pretrained=None,
        num_clusters=FLAGS.num_clusters,
        beam_limit=5,
        root_models=FLAGS.root_models,
        root_datasets=FLAGS.root_datasets,
        root_segmentations=FLAGS.root_segmentations,
        root_activations=FLAGS.root_activations,
        root_results=FLAGS.root_results,
        device=FLAGS.device
    )

    cfg_trained = settings.Settings(
        subset=FLAGS.subset,
        model=FLAGS.model,
        pretrained=FLAGS.pretrained,
        num_clusters=FLAGS.num_clusters,
        beam_limit=5,
        root_models=FLAGS.root_models,
        root_datasets=FLAGS.root_datasets,
        root_segmentations=FLAGS.root_segmentations,
        root_activations=FLAGS.root_activations,
        root_results=FLAGS.root_results,
        device=FLAGS.device

    )
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
    trained_model = model_utils.load_model_from_settings(
        cfg_trained, device=cfg.device)

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

        selected_units = range(11)
        all_labels = []
        for unit in tqdm(
            selected_units,
            desc="Loading Explanations for the untrained network"
        ):
            # Get activation ranges from clusters
            unit_activations = activations[unit]

            # Clustered compositional explanations ranges
            activation_ranges = activation_utils.compute_activation_ranges(
                unit_activations, FLAGS.num_clusters)

            for cluster_index, activation_range in enumerate(
                activation_ranges
            ):
                dir_current_results = (
                    f"{cfg.get_results_directory()}/"
                    + f"{layer_name}/{unit}/{activation_range}"
                )
                file_algo_results = (
                    f"{dir_current_results}/" + f"3.pickle"
                )
                try:
                    with open(file_algo_results, "rb") as file:
                        best_label, _, _ = pickle.load(file)
                except FileNotFoundError as exc:
                    raise FileNotFoundError(
                        f"File {file_algo_results} not found. You have to run "
                        + "the run_clustering script first "
                        + "using the flag --pretrained=none."
                    ) from exc
                all_labels.append(best_label)

        summary = [
            (F.get_formula_str(k, dataset.labels), v)
            for k, v in Counter(all_labels).most_common()
            if v > 1
        ]
        print(f"Summary (filtering out labels with count=1): {summary}")
        summary = [
            (label, count) for label, count
            in Counter(all_labels).most_common()
            if count > 1
        ]

        # Compute default formulas
        top_k = math.ceil((len(selected_units)/100)*20)
        default_formulas = [
            label for label, count
            in Counter(all_labels).most_common(top_k)]
        default_two = []

        index = 0
        while len(default_two) < top_k and index < len(summary):
            candidate = summary[index][0].left
            if candidate not in default_two:
                default_two.append(candidate)
            index += 1

        # Get activations
        activations = model_utils.get_layer_activations(
            segmentation_loader,
            trained_model,
            layer_name,
            range(num_units),
            cfg_trained.get_activation_directory(),
        )

        default_ranges = [0 for _ in range(FLAGS.num_clusters)]
        semi_default_ranges = [0 for _ in range(FLAGS.num_clusters)]
        for unit in tqdm(
            selected_units, desc="Loading Explanations for the trained network"
        ):
            # Get activation ranges from clusters
            unit_activations = activations[unit]

            # Clustered compositional explanations ranges
            activation_ranges = activation_utils.compute_activation_ranges(
                unit_activations, FLAGS.num_clusters)

            for cluster_index, activation_range in enumerate(
                activation_ranges
            ):
                dir_current_results = (
                    f"{cfg_trained.get_results_directory()}/"
                    + f"{layer_name}/{unit}/{activation_range}"
                )
                file_algo_results = (
                    f"{dir_current_results}/" + f"3.pickle"
                )
                try:
                    with open(file_algo_results, "rb") as file:
                        best_label, _, _ = pickle.load(file)
                except FileNotFoundError as exc:
                    raise FileNotFoundError(
                        f"File {file_algo_results} not found. You have to run "
                        + "the run_clustering script first "
                        + f"using the flag --pretrained={FLAGS.pretrained}."
                    ) from exc
                if best_label in default_formulas:
                    default_ranges[cluster_index] += 1
                elif best_label.left in default_two:
                    semi_default_ranges[cluster_index] += 1
        for cluster_index in range(FLAGS.num_clusters):
            percentage_default_len3 = \
                default_ranges[cluster_index] / len(selected_units)
            percentage_default_len2 = \
                semi_default_ranges[cluster_index] / len(selected_units)
            print("Unspecialized activations for " +
                  f"cluster {cluster_index}: {percentage_default_len3:.2f}")
            print("Weakly specialized activation for " +
                  f"cluster {cluster_index}: {percentage_default_len2:.2f}")


if __name__ == "__main__":
    with torch.no_grad():
        absl.app.run(main)
