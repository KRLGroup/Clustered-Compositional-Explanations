Official repository of the paper "Towards a fuller understanding of neurons with Clustered Compositional Explanations. Biagio La Rosa, Leilani Gilpin, and Roberto Capobianco and Daniele Nardi. NeurIPS 2023"

The repository contains the PyTorch code to replicate paper results and a guide to use Clustered Compositional Explanations in your own projects.

========= HEURISTIC =========

To use the MMESH heuristic to estimate the IOU import it and use provide the following arguments 
```
import heuristics
estimate_iou = heuristics.mmesh_heuristic(formula, heuristic_info, num_hits, MAX_SIZE_MASK)
```
where:
- *formula*: is the formula for which the iou must be estimated. It must be an istance of the formula.F classs
- *heuristic_info*: It is a nested tuple (dissect_info, enneary_info) where 
    - *dissect_info* info is a tuple containing information computed before and during the first beam. It is a nested tuple (unary_info, neuron_areas, unary_intersection) where
        - *unary_info* is a tuple containing the information for each atomic concepts extracted from the concepts dataset
            - *unary_areas*: is the size of the concept per sample
            - *unary_inscribed*: are the coordinates (top left and bottom right) of the rectangles inscribed in the concept's mask per sample
            - *unary_bounding_box*: are the coordinates (top left and bottom right) of the bounding box around the concept's mask per sample
        - *neuron_areas* is a vector containing the number of hits in the neuron's bitmap per sample
        - *unary_intersection* is a vector of size num_concepts containing the number of hits per sample in the masks obtained by the AND of the neuron's bitmap and the atomic concepts' mask 
    - *enneary_info* isa tuple containing the information collected for each formula in the current beam during the parsing of the previous beam. 
        - enneary_areas:  is the size of the formula's mask per sample.
        - enneary_inscribed: are the coordinates (top left and bottom right) of the rectangles inscribed in the formula's mask per sample
        - enneary_bounding_box: are the coordinates (top left and bottom right) of the bounding box around the formula's mask per sample
        - *enneary_intersection* is a vector containing the number of hits per sample in the masks obtained by the AND of the neuron's bitmap and the formula's mask
- *num_hits*: is the number of 1s in the neuron's bitmap
- *MAX_SIZE_MASK*: is the size of the mask including both 0 and 1 (i.e., number of cells)

========= SCRIPTS =========

All the scripts assume that you have downloaded the Broden dataset. You can use the script `dlbroden.sh` to do it. There are additional parameters that can be changed (like the layers to consider for each model) in the `constants.py` and `settings.py` files.

**run_clustering.py** 

It is the main script to run the Clustered Compositional Explanations algorithm. Most of the scripts require to run this script before calling them.
```
run_clustering.py --subset=ade20k --model=resnet18 --pretrained=places365 --heuristic=mmesh --length=3 --beam_limit=5 --num_clusters=5 --device=device --random_units=0 --root_models=data/model/ --root_datasets=data/dataset/ --root_segmentations=data/cache/segmentations/ --root_activations=data/cache/activations/ --root_results=data/results/ --seed=0
```
where: 
- *subset*: the concept dataset used to extract the semantic for each neuron. Admissible values are:[ade20k, pascal]
- *model*: the model to probe. Supported models are: [resnet18, vgg16, alexnet, densenet161]
- *heuristic*: heuristic to use in the beam search. Admissible values are [mmesh, areas, cfh, none]
- *pretrained*: the weights used to inizialize the model. Admissible values are:[place365, imagenet, none]
- *length*: Explanations' length and length of the beam search. Length=1 corresponds to NetDissect. Length=3 corresponds to Compositional Explanations and Clustered Compositional Explanations
- *num_clusters* : number of clusters to use to clusters each neuron's activation map. num_clusters=1 corresponds to NetDissect and Compositional Explanations and only the highest activations will be considered. num_clusters=5 corresponds to the results reported in Clustered Compositional Explanations paper.
- *beam_limit*: widenesss of the beam search. How many candidates to consider for each step. Note that the first step selects beam_limit*2 candidates among the unary concepts.
- *device*: device used to store the data. GPU is strongly recommended. Admissible values are:[cuda, cpu]
- *random_units*: number of units for which to compute the explanations. Set random_units to 0 to run the algorithm for all the units.
- *root_models*: Directory where to store/load the models
- *root_datasets*: Directory from where to load the dataset
- *root_segmentations*: Directory where to store/load the (sparse) concepts' segmentations
- *root_activations*: Directory where to store/load the units' activations
- *root_results*: Directory where to store/load the results
- *seed*: seed used to fix the randomness

**compare_heuristics.py**

It prints the timing and the visited states of the selected heuristics (mmesh, cfh, none ). The description of the parameters is the same of the ones of the `run_clustering.py` script.


```
compare_heuristics.py --heuristic=<HEURISTIC> --subset=ade20k --model=resnet18 --pretrained=places365  --length=3 --beam_limit=5 --num_clusters=1 --device=device --random_units=100 --root_models=data/model/ --root_datasets=data/dataset/ --root_segmentations=data/cache/segmentations/ --root_activations=data/cache/activations/ --root_results=data/results/ --seed=0
```
**compare_thresholds.py**

It compares explanations when the quantile is set to one of the following values [0.005, 0.01, 0.05, 0.1, 0.2, 0.5, 0.8, 0.9, 0.95, 0.99, 0.995]. The description of the parameters is the same of the ones of the `run_clustering.py` script.
```
compare_thresholds.py --subset=ade20k --model=resnet18 --pretrained=places365 --heuristic=mmesh --length=3 --device=device --random_units=0 --root_models=data/model/ --root_datasets=data/dataset/ --root_segmentations=data/cache/segmentations/ --root_activations=data/cache/activations/ --root_results=data/results/ --seed=0
```

**compute_metrics.py**

It calculates and prints metrics for the explanations returned by the `run_clustering.py` script. It requires to run the `run_clustering.py` script using the same parameters before running this script.

```
compute_metrics.py --subset=ade20k --model=resnet18 --pretrained=places365 --heuristic=mmesh --length=3 --beam_limit=5 --num_clusters=5 --device=device --random_units=0 --root_models=data/model/ --root_datasets=data/dataset/ --root_segmentations=data/cache/segmentations/ --root_activations=data/cache/activations/ --root_results=data/results/ --seed=0
```


**compute_ranges_importance.py**

It computes how many times the model changes its predictions when the activations associated with each cluster are masked.

```
compute_ranges_importance.py --subset=ade20k --model=resnet18 --pretrained=places365 --num_clusters=5 --device=device --random_units=0 --root_models=data/model/ --root_datasets=data/dataset/ --root_segmentations=data/cache/segmentations/ --root_activations=data/cache/activations/ --root_results=data/results/ --seed=0
```

**analyze_activations.py** 

This script can be used to analyze the type of activations inside each cluster (unspecialized and weakly specialized).
This script requires to run both `run_clustering.py --length=3 <REST_OF_PARAMETERS>` and  `run_clustering.py --pretrained=none --length=3 <REST_OF_PARAMETERS>` using the same parameters this script.
```
analyze_activations.py --subset=ade20k --model=resnet18 --pretrained=places365 --heuristic=mmesh --num_clusters=5 --device=device --random_units=0 --root_models=data/model/ --root_datasets=data/dataset/ --root_segmentations=data/cache/segmentations/ --root_activations=data/cache/activations/ --root_results=data/results/ --seed=0
```

**generate_images.py** 

This script can be used to generate a visual image of the clusters semantics for each neuron. Images are stored in the directory specified as a parameter using the following name `unit_{unit}_c_{num_clusters}.png`
This script requires to run the `run_clustering.py` script using the same parameters before running this script. 
```
generate_images.py --subset=ade20k  --model=resnet18 --pretrained=places365 --length=3 --num_clusters=5 --device=device 
--top_k=5 --random_units=0 --root_models=data/model/ --root_datasets=data/dataset/ --root_segmentations=data/cache/segmentations/ --root_activations=data/cache/activations/ --root_results=data/results/ --seed=0
```
where 
- *top_k* specify how many images should be printed for each cluster.

========= SET UP THE REPO =========
1) Install docker 

2) Pull the nvidia docker image pytorch:23.01-py3

3) Download the modified version of cocoapi from https://github.com/jayelm/cocoapi and save it in the workspace

4) Use the following dockerfile from the workspace to create a docker satisfying all the dependencies
```
# start from the nvidia docker image
FROM nvcr.io/nvidia/pytorch:23.01-py3

USER root
ARG UID=1000
ARG GID=1000
ARG DEBIAN_FRONTEND=noninteractive

# Remove pycocotools and install the modified version
RUN pip uninstall pycocotools -y
RUN mkdir /code
WORKDIR /code
ADD ./cocoapi /code
WORKDIR /code/PythonAPI
RUN python setup.py build_ext --inplace
RUN python setup.py build_ext install
WORKDIR /
RUN rm -r /code
WORKDIR /workspace

# install the remaining dependencies
RUN pip install pyeda seaborn imageio scikit-image
RUN pip uninstall pyparsing -y
RUN pip install pyparsing==2.4.2
```
5) Run the created image

6) Clone this repository

7) Run `dowload_models.sh` and `dlbroden.sh`

8) Run your scripts

Note that the first run will be slow since the repository generates and saves the sparse representations of the segmentation masks and computes and stores all the heuristic information for atomic concepts. Once that all these items are generated, the successive runs will be a lot faster.

========= DEPENDENCIES =========

This is the list of already tested packages' versions to succesfully run the scripts stored in this repo.
```
imageio=2.27.0
pyeda=0.28.0
scipy=1.9.1
seaborn=0.12.1
pytorch=1.14.0a0+44dac51
scikit-image=0.20.0
```

========= REPOSITORY REFERENCES =========

Here there is the list of repository and links used as a reference for the current repo

Compositional Explanations: https://github.com/jayelm/compexp/blob/master/vision/
Pytorch Randomness: https://pytorch.org/docs/stable/notes/randomness.html
NetDissect Lite: https://github.com/CSAILVision/NetDissect-Lite
Detection Accuracy: https://github.com/KRLGroup/detacc-compexp
cocoapi: https://github.com/jayelm/cocoapi
