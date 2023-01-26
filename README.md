# Video Object Detection
This program extracts and counts moving objects, e.g. people, cars and others, from a given sequence of frames (video) using Gaussian Mixture background modelling and detects pedestrians using Haar Cascades full body detector. The program assumes that the video frames are captured by a stationary camera.   

## Description

This Python program was developed for a computer vision subject at university with the purpose of exploring pedestrian/moving object detection due to its importance in intelligent video surveillance systems and autonomous driving. The program runs in the command line and where the user can choose whether they want the program to perform moving object detection (background modelling) or pedestrian detection and tracking. 

The moving object detection functionality uses Gaussian Mixture background modelling and substracts the estimated background in order to detect and extract moving pixels. Noise is removed before performing connected component analysis where blobs (or collections of foreground pixels) are extracted. These blobs are the moving objects. Simple classification of the moving objects into "person", "car", or "other" is then performed based on their aspect ratio (width/height). The reason for this simplistic classification method is that the focus of the task was on detection and extraction of moving objects and not on the classification itself. While running the background modelling functions on each frame of the specified video, the following is displayed in a single window:
    - original frame
    - estimated background frame
    - detected moving pixels before filtering
    - detected objects

In addition, the frame number and number of identified objects is printed. 


## Getting Started

### Dependencies
- Python 3
- OpenCV 4.5.5

## Executing the Program
