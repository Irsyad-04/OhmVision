
Deteksi Nilai Resistor - v1 2026-05-20 8:13pm
==============================

This dataset was exported via roboflow.com on May 21, 2026 at 1:25 AM GMT

Roboflow is an end-to-end computer vision platform that helps you
* collaborate with your team on computer vision projects
* collect & organize images
* understand and search unstructured image data
* annotate, and create datasets
* export, train, and deploy computer vision models
* use active learning to improve your dataset over time

For state of the art Computer Vision training notebooks you can use with this dataset,
visit https://github.com/roboflow/notebooks

To find over 100k other datasets and pre-trained models, visit https://universe.roboflow.com

The dataset includes 4008 images.
Object-resistors are annotated in YOLOv11 format.

The following pre-processing was applied to each image:
* Auto-orientation of pixel data (with EXIF-orientation stripping)
* Resize to 640x640 (Fit within)

The following augmentation was applied to create 3 versions of each source image:
* 50% probability of horizontal flip
* Randomly crop between 0 and 5 percent of the image
* Random rotation of between -15 and +15 degrees
* Random brigthness adjustment of between -10 and +10 percent
* Random exposure adjustment of between -5 and +5 percent
* Random Gaussian blur of between 0 and 1.3 pixels
* Salt and pepper noise was applied to 0.5 percent of pixels


