"""Module containing the class for the settings.
It is an adaptation of the code referenced in:
https://github.com/jayelm/compexp/blob/master/vision/settings.py"""

import torch

class Settings:
    """
    Class that stores all the settings used in each run.
    """

    def __init__(
        self,
        *,
        subset,
        model,
        pretrained,
        num_clusters,
        beam_limit,
        device,
        dataset="places365",
        root_models="data/model/",
        root_datasets="data/dataset/",
        root_segmentations="data/cache/segmentations/",
        root_activations="data/cache/activations/",
        root_results="data/results/",
    ):
        self.index_subset = subset
        self.model = model
        self.dataset = dataset
        if pretrained == "places365" or pretrained == "imagenet":
            self.pretrained = pretrained
        else:
            self.pretrained = None
        self.num_clusters = num_clusters
        self.beam_limit = beam_limit
        self.dir_datasets = root_datasets
        self.__root_segmentations = root_segmentations
        self.__root_activations = root_activations
        self.__root_results = root_results
        self.__root_models = root_models
        self.set_device(device)

    def set_device(self, device):
        """
        Sets the device to be used.
        """
        if "cuda" in device:
            if torch.cuda.is_available():
                self.device = torch.device(device)
            else:
                raise ValueError(
                    f"Device {device} not available. "
                    "Please use cpu or check your drivers."
                )
        elif "cpu" in device:
            self.device = torch.device(device)
        elif device is None:
            self.device = torch.device("cpu")
        else:
            raise ValueError(
                f"Device {device} not recognized. "
                "Please use cuda or cpu."
            )

    def get_image_mean(self):
        """
        Returns the mean of the dataset.
        """
        if self.pretrained == "imagenet":
            return [0.485, 0.456, 0.406]
        elif self.pretrained == "places365":
            return [0.485, 0.456, 0.406]
        else:
            return [0.5, 0.5, 0.5]

    def get_image_stdev(self):
        """
        Returns the standard deviation of the dataset.
        """
        if self.pretrained == "imagenet":
            return [0.229, 0.224, 0.225]
        elif self.pretrained == "places365":
            return [0.229, 0.224, 0.225]
        else:
            return [0.5, 0.5, 0.5]

    def get_num_classes(self):
        """
        Returns the number of classes of the dataset.
        """
        if self.dataset == "places365":
            return 365
        elif self.dataset == "imagenet":
            return 1000
        elif self.dataset == "ade20k":
            return 365

    def get_model_file_path(self):
        """
        Returns the path to the pretrained weights of the model.
        """
        if self.pretrained == "places365":
            if self.model == "densenet161":
                model_file_name = (
                    "whole_densenet161_places365_python36.pth.tar"
                )
            else:
                model_file_name = f"{self.model}_places365.pth.tar"
            return self.__root_models + "zoo/" + model_file_name
        elif self.pretrained == "imagenet":
            return None
        else:
            return "<UNTRAINED>"

    def get_weights(self):
        """
        Returns the pretrained weights of the model.
        """
        if self.pretrained == "imagenet":
            return "IMAGENET1K_V1"
        elif self.pretrained == "places365":
            return self.get_model_file_path()
        else:
            return None

    def get_data_directory(self):
        """
        Returns the directory where the data is stored.
        """
        if self.model != "alexnet":
            return f"{self.dir_datasets}broden1_224"
        else:
            return f"{self.dir_datasets}broden1_227"

    def get_model_root(self):
        """
        Returns the root directory where the models are stored.
        """
        return self.__root_models

    def get_img_size(self):
        """
        Returns the size of the images.
        """
        if self.model != "alexnet":
            return 224
        else:
            return 227

    def get_mask_shape(self):
        """
        Returns the shape of the mask.
        """
        if self.model != "alexnet":
            return (112, 112)
        else:
            return (113, 113)

    def get_max_mask_size(self):
        """
        Returns the maximum size of the mask.
        """
        return self.get_mask_shape()[0] * self.get_mask_shape()[1]

    def get_feature_names(self):
        """
        Returns the names of the layers that will be used to
        extract the features.
        """
        if self.model == "resnet18":
            return [
               "layer4"
            ]
        elif self.model == "resnet50":
            return ["layer4"]
        elif self.model == "resnet101":
            return ["layer4"]
        elif self.model == "densenet161":
            return ["features"]
        elif self.model == "alexnet":
            return ["features"]
        elif self.model == "vgg16":
            return ["features"]

    def get_parallel(self):
        """
        Returns True if the model is parallelized.
        """
        if self.dataset == "places365":
            return True
        else:
            return False

    def get_segmentation_directory(self):
        """
        Returns the directory where the sparse segmentations
        are stored.
        """
        return (
            f"{self.__root_segmentations}{self.dataset}/"
            f"{self.index_subset}/{self.get_img_size()}/sparse"
        )

    def get_activation_directory(self):
        """
        Returns the directory where the activations are stored.
        """
        return (
            f"{self.__root_activations}{self.model}/"
            f"{self.index_subset}/"
            f"{'w_'+self.pretrained if self.pretrained else 'untrained'}"
        )

    def get_results_directory(self):
        """
        Returns the directory where the results are stored.
        """
        return (
            f"{self.__root_results}{self.dataset}/{self.index_subset}/"
            f"{'w_'+self.pretrained if self.pretrained else 'untrained'}/"
            f"{self.model}"
        )

    def get_info_directory(self):
        """
        Returns the directory where the info files are stored.
        """
        return (
            f"{self.__root_segmentations}{self.dataset}/"
            f"{self.index_subset}/{self.get_img_size()}"
        )
