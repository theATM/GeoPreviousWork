# geo-detection-notebook

This code is part of the Magister Thesis.
It is a fork of the https://github.com/Dr-Zhuang/geospatial-object-detection
a Tensorflow 1.x based Yolov3 framework for the remote sensing object detection.

Main improvement over the existing code is an addition of a Docker based training pipeline.



## Dockerfile
To run the training process on modern gpu (RTX 3060 with CUDA 11.7) one needs to install docker image with old versions of the Tensorflow lib.
A dockerfile should contain those six lines of code (found in the docker/dockerNvi file):

* FROM nvcr.io/nvidia/tensorflow:22.11-tf1-py3
* RUN pip install numpy
* RUN pip install keras-applications==1.0.8
* RUN pip install keras-preprocessing==1.1.2
* RUN pip install keras==2.2.4
* RUN pip install opencv-python

to run the container additional settings are needed:

* --entrypoint -v /...projectpath.../ geo-detection-notebook:/opt/project -v /...datapath.../RSD-GOD:/opt/project/datasets:ro --rm --gpus all

## Train/Test/Predict

CPU based testing and prediction (from checkpoint or from pretrained weights - avaiable in the geospatial-object-detection repo)
can be accomplished without a fus, therefore a requirementsTC.txt file is presented to dowload the nesesary libs, as the conda environment was used.

For the CPU only testing/prediciton one need to use the config_cpu.json for the docker based training the config.json is used.

Prediction (predict.py) args:
* -c config_cpu.json -i RSD-GOD/testingsets/image/ -o predicts/ -m model/trained_model.h5
* (-c config_path -i input_images -o output_images -m model_used_to_predict)

Evaluation (evaluate.py) args:
* -c config_cpu.json -m model/trained_model.h5
* (-c config_path -m model_used_for_evaluation)

Training (train.py) args (in addition to the docker run args):
* -c config.json
* (-c config_path)


Initially the Google coolab code was intedned to be added, but coolab stopoed supporting the tensorflow 1.x pipeline.
Nevertheless, the main.ipynb is a working coolab code  (dataset needs to be on google drive) that is able to predict on dowloaded weights.

## Dataset

Please insert dataset (works with the RSD-GOD dataset) with specified structure (or change it in the config.json file):

RSD-GOD:
* testingsets
  * image
    * images.jpg
  * xml
    * labels.xml (VOC format)
* trainingsets
  * image
    * images.jpg
  * xml
    * labels.xml (VOC format)
* validationsets
  * image
    * images.jpg
  * xml
    * labels.xml (VOC format)


# geospatial-object-detection original README:
The code is mainly derived from [experiencor](https://github.com/experiencor/keras-yolo3).

Installs:
to use newer gpu install :
(used with python 3.8)
nvidia-pyindex==1.0.5
nvidia-tensorflow[horovod] (and no tensorflow-gpu==1.15.5)
or use cpu



# Paper：
Zhuang S, Wang P, Jiang B, et al. A Single Shot Framework with Multi-Scale Feature Fusion for Geospatial Object Detection[J]. Remote Sensing, 2019, 11(5): 594.

Considering the practical applications especially in military field, five categories are selected to be annotated, including plane, helicopter, oiltank, airport and warship. Finally, we construct a large-scale remote sensing dataset for geospatial object detection, which totally contains 18,187 color images with multiple resolutions from multiple platforms like Google Earth. There are 40,990 well-annotated instances in the dataset. The width of each image is mostly about 300~600 pixels. To increase the diversity of samples, we collect these remote-sensing images from different places at different times. 

## RSD-GOD dataset
The RSD-GOD dataset is avaliable at: [Google drive](https://drive.google.com/open?id=1ttvSta0BRxW7tTV_st89vSb_obHVre34);
also avaliable at: [baiduyun](https://pan.baidu.com/s/11J6n-CoMQ_EtFdx_KUs4PA).
