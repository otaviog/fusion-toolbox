FROM nvidia/cudagl:10.0-devel-ubuntu18.04
MAINTAINER otaviog

RUN apt update
RUN apt -yq install python3 aria2

RUN aria2c https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
RUN bash Miniconda3-latest-Linux-x86_64.sh -b
ENV PATH="/root/miniconda3/bin/:${PATH}"

#RUN conda install pytorch torchvision cudatoolkit=10.0 -c pytorch
#RUN conda install glew glfw -c conda-forge
#RUN conda install cmake eigen
#RUN conda install cudnn

ADD . /slam-toolbox

WORKDIR /slam-toolbox

RUN conda env update -n base --file environment.yml
RUN conda install cudnn -c pytorch

RUN apt -yq install libboost-filesystem-dev
RUN python setup.py install