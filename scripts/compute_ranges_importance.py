"""
This script computes the number of changed predictions when masking the
activations of a layer. This is done for each cluster of the compositional
explanation.
"""
import random

import torch
import torchvision
import absl.flags
import absl.app
from tqdm import tqdm

from src import segmentations
from src import model_utils
from src import activation_utils
from src import utils
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


def mask_activations(loader, model, layer_name, units, ranges, num_clusters):
    """
    This script masks the activations of a layer and computes the change in the
    prediction. This is done for each cluster of the compositional explanation.

    Args:
        loader: dataloader to use to load the data
        model: model to use to compute the prediction
        layer_name: name of the layer to mask
        units: units to mask
        ranges: ranges to use to mask the units
        num_clusters: number of clusters to use

    Returns:
        changed_prediction: number of changed predictions per cluster
    """
    device = next(model.parameters()).device
    temp_output = [[] for _ in range(num_clusters)]
    active_in = [[] for _ in range(num_clusters)]

    def hook_feature(module, inp, hidden):
        for cluster_indx in range(num_clusters):
            modified_hidden = hidden.clone()

            for index, unit in enumerate(units):
                lower_thereshold = ranges[index][cluster_indx][0]
                high_threshold = ranges[index][cluster_indx][1]
                modified_hidden[:, unit] = torch.where(
                    (hidden[:, unit] > lower_thereshold) & (
                        hidden[:, unit] < high_threshold),
                    0.0,
                    hidden[:, unit],
                )
            modified_output = model.fc(
                torch.flatten(model.avgpool(modified_hidden), 1)
            )
            temp_output[cluster_indx].append(modified_output)

    handles = model_utils.hook(model, hook_feature, [layer_name])

    changed_prediction = [0 for _ in range(num_clusters)]

    for _, (images, _, _) in enumerate(loader):
        # Move to GPU
        images = images.to(device)

        # Forward pass
        outputs = model(images)

        # collect data
        for cluster_indx in range(num_clusters):
            out_softmax = torch.nn.functional.log_softmax(outputs, dim=1)
            original_prediction = torch.argmax(out_softmax, dim=1)
            modified_softmax = torch.nn.functional.log_softmax(
                temp_output[cluster_indx][0], dim=1
            )
            modified_prediction = torch.argmax(modified_softmax, dim=1)
            changed_prediction[cluster_indx] += torch.sum(
                original_prediction != modified_prediction
            )

        # Empty the temp list
        del temp_output[:]
        temp_output = [[] for _ in range(num_clusters)]
        del active_in[:]
        active_in = [[] for _ in range(num_clusters)]
    for handle in handles:
        handle.remove()
    return changed_prediction


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
        device=FLAGS.device,
        root_models=FLAGS.root_models,
        root_datasets=FLAGS.root_datasets,
        root_segmentations=FLAGS.root_segmentations,
        root_activations=FLAGS.root_activations,
        root_results=FLAGS.root_results,
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
        batch_size=C.BATCH_SIZE,
        worker_init_fn=utils.seed_worker,
        generator=generator,
    )

    # Load Model
    model = model_utils.load_model_from_settings(cfg, device=cfg.device)

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

        total_activation_ranges = []
        for unit in tqdm(
            selected_units, desc="Computing activation ranges per unit"
        ):
            unit_activations = activations[unit]

            # Clustered compositional explanations ranges
            activation_ranges = activation_utils.compute_activation_ranges(
                unit_activations, FLAGS.num_clusters)

            total_activation_ranges.append(activation_ranges)
        changed_prediction = mask_activations(
            segmentation_loader,
            model,
            [layer_name],
            units=selected_units,
            ranges=total_activation_ranges,
            num_clusters=FLAGS.num_clusters,
        )
        for cluster_indx in range(FLAGS.num_clusters):
            cluster_score = changed_prediction[cluster_indx]/len(
                segmentation_loader.dataset)
            print(f"Cluster {cluster_indx} Pred Changes: {cluster_score}")


if __name__ == "__main__":
    with torch.no_grad():
        absl.app.run(main)
