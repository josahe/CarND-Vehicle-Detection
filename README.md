# Vehicle Detection and Tracking
> An image processing pipeline to detect and track vehicles

## INTRODUCTION
The goal of this project is to write a software pipeline that detects vehicles in an image and tracks them across frames of a video captured from a front-facing camera on a car

A demonstration of the pipeline is show in this [video](project_video_output.mp4)

This project was undertaken as part of the [Udacity Self-Driving Car NanoDegree](https://eu.udacity.com/course/self-driving-car-engineer-nanodegree--nd013).

### Pipeline summary
* Extract features from a labelled training set of images and train a Linear SVM classifier
  * Features: histogram of oriented gradients, colour histograms and spatial binning
* Implement a sliding-window across an image and use a trained classifier to search for vehicles
* Create a heat map of recurring detections frame by frame to reject outliers and track detected vehicles
* Estimate a bounding box for each tracked vehicle

## HOW TO USE
### Project dependencies
You can follow the guide explained here to setup a working environment.
* https://github.com/udacity/CarND-Term1-Starter-Kit

### Jupyter Notebooks
The included notebooks demonstrate how to use the project code.
* [vehicle_detection.ipynb](vehicle_detection.ipynb)
  * Demonstrates the end-to-end process, from the dataset, to training the classifier and processing images and video
* [notebooks/pipeline_breakdown.ipynb](notebooks/pipeline_breakdown.ipynb)
  * Breaks down the image processing pipeline into stages to visualise the process
* [notebooks/helper.ipynb](notebooks/helper.ipynb)
  * A collection of other useful snippets of code used along the way

## RELEVANT LINKS
#### Project writeup
* [writeup.md](writeup.md)

#### Original project repo
* https://github.com/udacity/CarND-Vehicle-Detection

## RELEVANT FILES
* [vehicle_tracking.py](vehicle_tracking.py), including:
  * A Vehicle class to represent tracked vehicle objects
  * A VehicleTracking class that implements the full processing pipeline
* [helper.py](helper.py), including independent implementations of various stages of the pipeline and other miscellaneous helper functions
