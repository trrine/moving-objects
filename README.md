# Video Object Detection
This program extracts and counts moving objects, e.g. people, cars and others, from a given sequence of frames (video) using Gaussian Mixture background modelling and detects pedestrians using Haar Cascades full body detector. The program assumes that the video frames are captured by a stationary camera.   

## Description

This Python program was developed for a computer vision subject at university with the purpose of exploring pedestrian/moving object detection due to its importance in intelligent video surveillance systems and autonomous driving. The program runs in the command line and where the user can choose whether they want the program to perform moving object detection and extraction (background modelling) or pedestrian detection and tracking. 

The moving object detection functionality uses Gaussian Mixture background modelling and substracts the estimated background in order to detect and extract moving pixels. Noise is removed before performing connected component analysis where blobs (or collections of foreground pixels) are extracted. These blobs are the moving objects. Simple classification of the moving objects into "person", "car", or "other" is then performed based on their aspect ratio (width/height). The reason for this simplistic classification method is that the focus of the task was on detection and extraction of moving objects and not on the classification itself. While running the background modelling functions on each frame of the specified video, the following is displayed in a single window:
- Original frame
- Estimated background frame
- Detected moving pixels before filtering
- Detected objects

In addition, the frame number and number of identified objects are printed. 

The pedestrian detection and tracking functionality uses the Haar Cascade full body detector to detect pedestrians in each video frame. It also tracks each detected pedestrian with a bounding box and a unique ID by finding the Euclidian distance between the center of the bounding box of a detected pedestrian and the center of bounding boxes of previously detected pedestrians and determining whether they are the same pedestrian based on a threshold or whether to give the detected pedestrian a new ID. In each frame, the program also finds the three pedestrians that are the closest to the camera based on a size/distance assumption. While running the pedestrian detection and tracking functionality on each frame, the following is displayed in a single window:
- Original frame
- Frame with overlapped detected bounding boxes
- Frame with detected and tracked (labelled) bounding boxes
- Frame with up to three detected objects closest to camera

## Getting Started

### Dependencies
- Python 3
- OpenCV 4.5.5

### Executing the Program

The program runs from the command line where the user inputs a video to the analysed. To run the program with the first functionality (moving object detection/background modelling), type the following command with the videofile you wish to use:
- movingObj.py –b videofile
- E.g. movingObj.py -b trafficlights.avi

To run the second functionality (pedestrian detection and tracking), type the following command with the videofile you wish to use:
- movingObj –p videofile
- E.g. movingObj.py -p trafficlights.avi

The program runs until it has looped through each frame or until the user pressed "Escape" or "q".
