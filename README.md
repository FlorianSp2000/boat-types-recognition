# Boat Types Detection with YOLO

This project focuses on detecting various types of boats in images and video in real-time using a custom-trained object detection model. The model used is [YOLOv10](https://arxiv.org/pdf/2405.14458) and can identify 10 different classes of boats.

## Table of Contents

- [Overview](#overview)
- [Training Data](#training-data)
- [Installation](#installation)
- [Usage](#usage)
  - [Training](#training)
  - [Detection](#detection)
- [Results](#results)

## Overview

This repository contains:
- A Jupyter Notebook (`train.ipynb`) used for training the model
- A Python script (`detect.py`) for performing detection on images and real-time video streams.

The model can detect the following boat classes:
- boat (includes smaller vessels such as motorboats, that are not covered by the classes below)
- buoy
- cruise ship
- ferry boat
- freight boat
- gondola
- inflatable boat
- kayak
- paper boat
- sailboat

## Training Data

The training data originates from [Kaggle](https://www.kaggle.com/datasets/kunalgupta2616/boat-types-recognition?select=boat-types-recognition) and was manually annotated using RoboFlow. The annotated dataset is available under this [link](https://app.roboflow.com/floriansdatalake/boat-types-detection/3).

## Installation

1. **Clone the repository:**

    ```bash
    git clone https://github.com/FlorianSp2000/boat-types-recognition.git
    cd boat-types-recognition
    ```

2. **Install the required packages:**

    Install the dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### Training

1. Open the `train.ipynb` notebook using Google Colab.
2. Follow the instructions in the notebook to set up the environment, upload data, and start the training process.
3. Once the training is complete, the trained model weights will be saved and can be downloaded for local use.

### Inference 

The `detect.py` script can be used to perform detection on images and video streams. If --save flag is set, results are saved under /evaluation_results

#### Example Usage

Image Detection

```bash
python detect.py models/yolov10-30ep-1.pt path/to/your/img.jpg --type image 
```

Video Detection 

```bash
python detect.py models/yolov10-30ep-1.pt path/to/your/video.mp4 --type video
```

## Results

Trained models were evaluated qualitatively on two videos for real-time detection. The videos were taken from the perspective of another ship (i.e., front view).


#### Example Detections

