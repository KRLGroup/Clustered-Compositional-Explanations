#!/usr/bin/env bash
set -e

# Start from parent directory of script
cd "$(dirname "$(dirname "$(readlink -f "$0")")")"

echo "Downloading resnet18, alexnet, densenet161, and vgg16 trained on Places365"
mkdir -p data/model/zoo
pushd data/model/zoo
wget --progress=bar \
   http://places2.csail.mit.edu/models_places365/resnet18_places365.pth.tar
wget --progress=bar \
    http://places2.csail.mit.edu/models_places365/alexnet_places365.pth.tar
wget --progress=bar \
    http://places2.csail.mit.edu/models_places365/whole_densenet161_places365_python36.pth.tar
popd
echo "done"