import os
import pickle
import random
import absl.flags
import absl.app

from tqdm import tqdm
import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt

from src import segmentations
from src import settings
from src import model_utils
from src import mask_utils
from src import activation_utils
from src import utils
from src import constants as C
from src import metrics
from src import formula as F

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
absl.flags.DEFINE_integer("top_k", 5, "top k samples")
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
absl.flags.DEFINE_string(
    "dir_images", "figures/units/", "Where to save the images"
)
absl.flags.DEFINE_integer("seed", 0, "seed to use to set reproducibility")
FLAGS = absl.flags.FLAGS


def show(imgs, labels=None):
    """Show images in a grid"""
    if not isinstance(imgs, list):
        imgs = [imgs]
    fig, axs = plt.subplots(
        nrows=len(imgs), squeeze=False,
        gridspec_kw={'wspace': 0, 'hspace': 0.5})
    for i, img in enumerate(imgs):
        img = img.detach()
        img = torchvision.transforms.functional.to_pil_image(img)
        axs[i, 0].imshow(np.asarray(img))
        axs[i, 0].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
        if labels is not None:
            axs[i, 0].set_title(labels[i])
    return fig


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

    # Load data without normalization
    image_dataset = segmentations.BrodenDataset(
                cfg.dir_datasets,
                subset=cfg.index_subset,
                resolution=cfg.get_img_size(),
                broden_version=1,
                transform_image=torchvision.transforms.Compose([
                    torchvision.transforms.Resize(112)])
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
        sparse_segmentation_directory, segmentation_loader, dataset.labels,
        cfg.device
    )

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
        with tqdm(
                selected_units,
                desc="Computing Compostional explanations per unit") as pbar:
            for unit in pbar:
                # Get unit activations
                unit_activations = activations[unit]
                activation_ranges = activation_utils.compute_activation_ranges(
                    unit_activations, FLAGS.num_clusters)
                # Loop over all the activation ranges
                images_list = []
                labels_list = []
                for _, activation_range in enumerate(
                    sorted(activation_ranges)
                ):
                    bitmaps = activation_utils.compute_bitmaps(
                            unit_activations,
                            activation_range,
                            mask_shape=mask_shape,
                        )
                    bitmaps = bitmaps.to(cfg.device)

                    dir_current_results = (
                        f"{cfg.get_results_directory()}/"
                        + f"{layer_name}/{unit}/{activation_range}"
                    )
                    file_algo_results = (
                        f"{dir_current_results}/" + f"{FLAGS.length}.pickle"
                    )

                    # Load results
                    with open(file_algo_results, "rb") as file:
                        best_label, best_iou, _ = pickle.load(file)

                    # Filter the top k candidates samples that
                    label_mask = mask_utils.get_formula_mask(
                        best_label, masks).to(FLAGS.device)
                    # - contain the concept
                    samples_formula = label_mask.sum(1) > 0
                    # - have the neuron firing
                    neuron_fires = bitmaps.sum(1) > 0
                    # - have a high iou
                    samples_iou = metrics.sample_iou(bitmaps, label_mask)
                    above_iou = samples_iou > best_iou
                    candidates = neuron_fires & samples_formula & above_iou
                    nonzero = torch.nonzero(candidates).flatten()
                    top_k = random.sample(nonzero.tolist(), FLAGS.top_k)

                    # Plot the top k samples
                    
                    if not os.path.exists(FLAGS.dir_images):
                        os.makedirs(FLAGS.dir_images)
                    images = []
                    for index_sample in top_k:
                        data, _, _ = image_dataset[index_sample]
                        image = torch.from_numpy(np.array(data))
                        image = image.permute(2, 0, 1)
                        mask_concept = ~bitmaps[index_sample]
                        mask_concept = mask_concept.reshape(
                            mask_shape[0], mask_shape[1])
                        segmented_image = torchvision.utils.draw_segmentation_masks(
                            image, mask_concept, alpha=1, colors='black')
                        images.append(segmented_image)
                    images_list.append(
                        torchvision.utils.make_grid(
                            images, padding=2, pad_value=255)
                        )
                    title = f"{F.get_formula_str(best_label, dataset.labels)}"
                    labels_list.append(title)
                fig = show(images_list, labels_list)
                fig.set_size_inches(6, 7)
                fig.savefig(
                    f'{FLAGS.dir_images}/' +
                    f'unit_{unit}_c_{FLAGS.num_clusters}.png')
                pbar.update(0)


if __name__ == "__main__":
    absl.app.run(main)
