Official repository of the paper "**Towards a fuller understanding of neurons with Clustered Compositional Explanations**. *Biagio La Rosa, Leilani Gilpin, and Roberto Capobianco.* NeurIPS 2023"

The repository contains the PyTorch code to replicate paper results and a guide to use Clustered Compositional Explanations in your own projects.

## ========= HEURISTIC =========

To use the MMESH heuristic to estimate the IoU score import the `heuristic` package and provide the following arguments to the function:
```
import heuristics
estimate_iou = heuristics.mmesh_heuristic(formula, heuristic_info, num_hits, MAX_SIZE_MASK)
```
where:
- *formula*: is the formula for which the iou must be estimated. It must be an instance of the formula (F class)
- *heuristic_info*: It is a nested tuple (dissect_info, enneary_info) where 
    - *dissect_info* info is a tuple containing information computed before and during the first beam. It is a nested tuple (unary_info, neuron_areas, unary_intersection) where
        - *unary_info* is a tuple containing the information for each atomic concept extracted from the concepts dataset
            - *unary_areas*: is the size of the concept per sample
            - *unary_inscribed*: are the coordinates (top left and bottom right) of the rectangles inscribed in the concept's mask per sample
            - *unary_bounding_box*: are the coordinates (top left and bottom right) of the bounding box around the concept's mask per sample
        - *neuron_areas* is a vector containing the number of hits in the neuron's bitmap per sample
        - *unary_intersection* is a vector of size num_concepts containing the number of hits per sample in the masks obtained by the AND of the neuron's bitmap and the atomic concepts' mask 
    - *enneary_info* is a tuple containing the information collected for each formula in the current beam during the parsing of the previous beam. 
        - enneary_areas:  is the size of the formula's mask per sample.
        - enneary_inscribed: are the coordinates (top left and bottom right) of the rectangles inscribed in the formula's mask per sample
        - enneary_bounding_box: are the coordinates (top left and bottom right) of the bounding box around the formula's mask per sample
        - *enneary_intersection* is a vector containing the number of hits per sample in the masks obtained by the AND of the neuron's bitmap and the formula's mask
- *num_hits*: is the number of 1s in the neuron's bitmap
- *MAX_SIZE_MASK*: is the size of the mask including both 0 and 1 (i.e., number of cells)

## ========= SCRIPTS =========

All the scripts are stored in the `scripts/` directory, assume that you have [downloaded the Broden](#download_broden) dataset, and assume you have [installed the package](#install_package) (needed). You can use the script `dlbroden.sh` to do it. Run them from the parent directory. There are additional parameters that can be changed (like the layers to consider for each model) in the `constants.py` and `settings.py` files.

**scripts/run_clustering.py** 

It is the main script to run the Clustered Compositional Explanations algorithm. Most of the scripts require running this script before calling them.
```
python3 scripts/run_clustering.py --subset=ade20k --model=resnet18 --pretrained=places365 --length=3 --beam_limit=5 --num_clusters=5 --device=device --random_units=0 --root_models=data/model/ --root_datasets=data/dataset/ --root_segmentations=data/cache/segmentations/ --root_activations=data/cache/activations/ --root_results=data/results/ --seed=0
```
where: 
- *subset*: the concept dataset used to extract the semantics for each neuron. Admissible values are:[ade20k, pascal]
- *model*: the model to probe. Supported models are: [resnet18, vgg16, alexnet, densenet161]
- *pretrained*: the weights used to initialize the model. Admissible values are:[place365, imagenet, none]
- *length*: Explanations' length and length of the beam search. Length=1 corresponds to NetDissect. Length=3 corresponds to Compositional Explanations and Clustered Compositional Explanations
- *num_clusters*: number of clusters to use to cluster each neuron's activation map. num_clusters=1 corresponds to NetDissect and Compositional Explanations and only the highest activations will be considered. num_clusters=5 corresponds to the results reported in the Clustered Compositional Explanations paper.
- *beam_limit*: wideness of the beam search. How many candidates to consider for each step. Note that the first step selects beam_limit*2 candidates among the unary concepts.
- *device*: device used to store the data. GPU is strongly recommended. Admissible values are:[cuda, cpu]
- *random_units*: number of units for which to compute the explanations. Set random_units to 0 to run the algorithm for all the units.
- *root_models*: Directory where to store/load the models
- *root_datasets*: Directory from where to load the dataset
- *root_segmentations*: Directory where to store/load the (sparse) concepts' segmentations
- *root_activations*: Directory where to store/load the units' activations
- *root_results*: Directory where to store/load the results
- *seed*: seed used to fix the randomness

**scripts/compare_heuristics.py**

It prints the timing and the visited states of the selected heuristics (mmesh, cfh, none ). The description of the parameters is the same as the ones of the `run_clustering.py` script.


```
python3 scripts/compare_heuristics.py --heuristic=<HEURISTIC> --subset=ade20k --model=resnet18 --pretrained=places365  --length=3 --beam_limit=5 --num_clusters=1 --device=device --random_units=100 --root_models=data/model/ --root_datasets=data/dataset/ --root_segmentations=data/cache/segmentations/ --root_activations=data/cache/activations/ --root_results=data/results/ --seed=0
```
where:
- *heuristic*: heuristic to use in the beam search. Admissible values are [mmesh, areas, cfh, none]

**scripts/compare_thresholds.py**

It compares explanations when the quantile is set to one of the following values [0.005, 0.01, 0.05, 0.1, 0.2, 0.5, 0.8, 0.9, 0.95, 0.99, 0.995]. The description of the parameters is the same of the ones of the `run_clustering.py` script.
```
python3 scripts/compare_thresholds.py --subset=ade20k --model=resnet18 --pretrained=places365 --length=3 --device=device --random_units=0 --root_models=data/model/ --root_datasets=data/dataset/ --root_segmentations=data/cache/segmentations/ --root_activations=data/cache/activations/ --root_results=data/results/ --seed=0
```

**scripts/compute_metrics.py**

It calculates and prints metrics for the explanations returned by the `run_clustering.py` script. It requires running the `run_clustering.py` script using the same parameters before running this script.

```
python3 scripts/compute_metrics.py --subset=ade20k --model=resnet18 --pretrained=places365 --length=3 --beam_limit=5 --num_clusters=5 --device=device --random_units=0 --root_models=data/model/ --root_datasets=data/dataset/ --root_segmentations=data/cache/segmentations/ --root_activations=data/cache/activations/ --root_results=data/results/ --seed=0
```


**scripts/compute_ranges_importance.py**

It computes how many times the model changes its predictions when the activations associated with each cluster are masked.

```
python3 scripts/compute_ranges_importance.py --subset=ade20k --model=resnet18 --pretrained=places365 --num_clusters=5 --device=device --random_units=0 --root_models=data/model/ --root_datasets=data/dataset/ --root_segmentations=data/cache/segmentations/ --root_activations=data/cache/activations/ --root_results=data/results/ --seed=0
```

**scripts/analyze_activations.py** 

This script can be used to analyze the type of activations inside each cluster (unspecialized and weakly specialized).
This script requires running both `run_clustering.py --length=3 <REST_OF_PARAMETERS>` and  `run_clustering.py --pretrained=none --length=3 <REST_OF_PARAMETERS>` using the same parameters this script.
```
python3 scripts/analyze_activations.py --subset=ade20k --model=resnet18 --pretrained=places365 --num_clusters=5 --device=device --random_units=0 --root_models=data/model/ --root_datasets=data/dataset/ --root_segmentations=data/cache/segmentations/ --root_activations=data/cache/activations/ --root_results=data/results/ --seed=0
```

**scripts/generate_images.py** 

This script can be used to generate a visual image of the clusters semantics for each neuron. Images are stored in the directory specified as a parameter using the following name `unit_{unit}_c_{num_clusters}.png`
This script requires running the `run_clustering.py` script using the same parameters before running this script. 
```
python3 scripts/generate_images.py --subset=ade20k  --model=resnet18 --pretrained=places365 --length=3 --num_clusters=5 --device=device 
--top_k=5 --random_units=0 --root_models=data/model/ --root_datasets=data/dataset/ --root_segmentations=data/cache/segmentations/ --root_activations=data/cache/activations/ --root_results=data/results/ --seed=0
```
where 
- *top_k* specify how many images should be printed for each cluster.

## ========= METRICS =========
The metrics described in the paper are stored inside the file `src/metrics`.
Here we provide an example of how to use them:
```
bitmaps = ... # Binary mask for the current neuron
label_mask = ... # Binary segmentation mask of the label associated with the neuron
from src import heuristics
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
explanation_coverage = metrics.explanation_coverage(
    bitmaps, label_mask)

```

Concept masking uses different arguments:
```
score = cosine_concept_masking_score(
    activation_before_masking, activation_after_masking, activation_range)
```
where:
- *activation_before_masking* are the activations of the neuron when fed with standard inputs
- *activation_after_masking* are the activations collected by masking all but the label's masks in inputs
- *activation_range* is the activation range considered

In order to compute concept masking without computing and providing the `activation_after_masking` you can use the auxiliary function:
```
concept masking = metrics.get_concept_masking(
                    unit=unit,
                    activations=unit_activations,
                    activation_range=activation_range,
                    label_mask=label_mask,
                    mask_shape=mask_shape,
                    layer_name=layer_name,
                    model=model,
                    loader=segmentation_loader,
                    input_size=image_size,
                )
```
where:
- *unit* is the index of the neuron
- *unit_activations* are the activations of the considered neuron
- *activation_range* is the activation range to consider to compute the label masking score
- *label_mask* is the mask of the labels associated with the current activation range
- *mask_shape* is the shape of a segmentation mask
- *layer_name* is the name of the layer where the neuron is placed
- *model* is the model where the neuron is placed
- *loader* is the loader of the concept dataset
- *input_size* is the size of the input (we assume squared input) 




## ========= SET UP THE REPO =========
1) Install docker 

2) Pull the nvidia docker image pytorch:23.01-py3

3) Clone this repository

4) Download the modified version of cocoapi from https://github.com/jayelm/cocoapi and save it in the "docker" folder inside the downloaded repository

5) Build the dockerfile from the docker directory 
```
docker -t <NAME_IMAGE>:<VERSION> .
```

6) Run the created image

7) Move to the repository 

8) <a name="download_broden"></a>Run `dowload_models.sh` and `dlbroden.sh`

9) <a name="install_package"></a>Install the package `pip install -e .`

10) Run your scripts

Note that the first run will be slow since the repository generates and saves the sparse representations of the segmentation masks and computes and stores all the heuristic information for atomic concepts. Once that all these items are generated, the successive runs will be a lot faster.

## ========= DEPENDENCIES =========

This is the list of already tested packages' versions to successfully run the scripts stored in this repo.
```
imageio=2.27.0
pyeda=0.28.0
scipy=1.9.1
seaborn=0.12.1
pytorch=1.14.0a0+44dac51
scikit-image=0.20.0
```

## ========= REPOSITORY REFERENCES =========

Here is the list of repositories and links used as a reference for the current repo

Compositional Explanations: https://github.com/jayelm/compexp/blob/master/vision/ <br>
Pytorch Randomness: https://pytorch.org/docs/stable/notes/randomness.html <br>
NetDissect Lite: https://github.com/CSAILVision/NetDissect-Lite <br>
Detection Accuracy: https://github.com/KRLGroup/detacc-compexp <br>
cocoapi: https://github.com/jayelm/cocoapi
