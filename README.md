---
title: AI Car Damage Detection System
emoji: 🚘
colorFrom: blue
colorTo: gray
sdk: gradio
sdk_version: 6.17.3
app_file: app.py
pinned: false
---

# Project Report: AI Car Damage Detection and Verification System

## Introduction and Purpose
The purpose of this project is to automate the vehicle damage inspection process by comparing two sets of photos representing the state of a vehicle before and after a rental period, insurance claim, or maintenance service. 

Traditional vehicle inspections are manual, subjective, and prone to disputes or fraud. By utilizing state-of-the-art computer vision models, this system validates the identity of the vehicle across views and identifies newly acquired damages (such as scratches, dents, or cracks on specific panels) while filtering out pre-existing ones.

## System Architecture
The application integrates three core vision pipelines:
1. **Vehicle Identity Verification**: Extracts visual feature embeddings using a pre-trained ResNet-18 network and calculates the cosine similarity between the reference and verification images to ensure they represent the same vehicle from equivalent angles.
2. **Homography-Based Perspective Alignment**: Detects keypoints and descriptors using ORB on the grayscale representations of both photos and computes a homography matrix to warp the "After" image to align with the "Before" reference perspective. This minimizes false positives caused by minor camera movement or shifts.
3. **Damage Detection and Filtering**: Processes both aligned images through a custom-trained YOLOv8s object detection model. It checks for bounding box overlaps using Intersection-over-Union (IoU). Any newly detected car part containing damage in the "After" state that was not present in the "Before" state is isolated and marked as new damage.

## Model Performance
The damage detection component is powered by a YOLOv8s model trained for 50 epochs. The validation performance metrics across the 7 target classes are summarized below:

| Class | Images | Instances | Precision (P) | Recall (R) | mAP50 | mAP50-95 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **All Classes** | 657 | 1325 | 0.574 | 0.502 | 0.499 | 0.264 |
| **Bonnet** | 162 | 165 | 0.764 | 0.594 | 0.676 | 0.367 |
| **Bumper** | 385 | 393 | 0.559 | 0.616 | 0.566 | 0.255 |
| **Dickey** | 57 | 57 | 0.517 | 0.509 | 0.457 | 0.237 |
| **Door** | 162 | 189 | 0.542 | 0.418 | 0.413 | 0.161 |
| **Fender** | 247 | 258 | 0.390 | 0.360 | 0.305 | 0.123 |
| **Light** | 173 | 193 | 0.479 | 0.451 | 0.410 | 0.218 |
| **Windshield** | 70 | 70 | 0.768 | 0.567 | 0.666 | 0.486 |

## Dataset Source
The model was trained on the capstone vehicle damage dataset version 4, hosted on Roboflow:
[Roboflow Dataset Link](https://universe.roboflow.com/capstone-nh0nc/car-damage-detection-t0g92/dataset/4)

## Live Application Demo
The application is deployed on Hugging Face Spaces:
[Hugging Face Space Live Demo](https://huggingface.co/spaces/chandimabandara/AI-Car-Damage-Detection-System)

## Visual Inspection Results
Below is the visual output showing the results of the inspection pipeline:

![Inspection Result](evidence/Screenshot%20from%202026-06-09%2014-32-03.png)
