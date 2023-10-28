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
