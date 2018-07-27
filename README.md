# Human-Pose-Classification
A Python and Keras implementation for classifying human arm poses

## About
Trained using the InceptionV3 model on a custom image dataset of 36 different positions.

## Installation
Requires the OpenPose Python API to be properly installed:
https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/doc/modules/python_module.md
Place this project in the compiled OpenPose directory e.g. 'C:/openpose-master/Release/python/openpose/this_project'
Download the model from ... and place it in the 'model' folder

## Usage
Run 'openpose.py' (on line 416 change use_webcam to true if you want to run through a webcam or image_dir to your own directory to run through a folder of images)

## Performance


## Credits
The human-pose estimation was made possible thanks to:
https://github.com/CMU-Perceptual-Computing-Lab/openpose
