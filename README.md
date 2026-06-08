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

# AI Car Damage Detection & Vehicle Verification System

An AI-powered web application that inspects vehicles for new damages by comparing "Before" and "After" state photos across 4 views (Front, Rear, Left, Right).

## Features
- **Vehicle Identity Matching**: Extracts ResNet-18 features to verify if the vehicle matches between photos and checks perspective/angle alignment.
- **Image Warping**: Uses OpenCV ORB descriptor homography to align "After" inspection images to "Before" reference images.
- **New Damage Localization**: Runs custom YOLOv8 model inference using a dual-confidence threshold comparison to isolate and draw bounding boxes around *new* damages only.
