# Previous Work 

This code is part of the Magister Thesis.
It is a fork of the https://github.com/Dr-Zhuang/geospatial-object-detection
a Tensorflow 1.x based Yolov3 framework for the remote sensing object detection using the RSD-GOD dataset.

The Main AirDetection repo can be found [under this link](https://github.com/theATM/AirDetection).

RSD-GOD dataset is a remote sensing dataset with obejcts such as aribases, helicopters, planes and warships.
Examples of detections (done on home trained model) can be seen on the figure bellow:

![rsd_detections_org_my](https://github.com/theATM/geo-detection-notebook/assets/48883111/b38234f3-36cc-4efc-b86f-f4a0754d90f7)




Main improvement over the existing code is an addition of a Docker based training pipeline.



## Dockerfile
This code works of cpu. To run the training process on modern gpu (RTX 3060 with CUDA 11.7) one needs to install docker image with old versions of the Tensorflow 1.15 lib  which is sadly not supported. A workaround is to use following dockerfile.
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

RSD-GOD [(download link)](https://drive.google.com/open?id=1ttvSta0BRxW7tTV_st89vSb_obHVre34):
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

## Training results
The model was trained on one gpu with batch size = 4. <br>
There are the detection results, mAP@50 on RSD-GOD dataset:

| Set  | All  | Airport   |  Helicopter  |  Oiltank  | Plane |  Warship |
|---|---|---|---|---|---|---|
| Val  | 0.8664  | 0.8388   |  0.9376  |  0.9217  | 0.8605 |  0.7732 |
| Test  | 0.7995  | 0.7102   |  0.8846  |  0.9029  | 0.8283 |  0.6716 |

Trained model O2 <a href="https://drive.google.com/file/d/1XbCKVi2A16a5E_rcWITa9IeLb5F1SGnt/view?usp=sharing">DOWNLOAD</a>

### Original README [Here](README_ORIGINAL.md)


