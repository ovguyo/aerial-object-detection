# Fine-tuning DETR for Aerial Object Detection (AUAIR Dataset) - Jupyter Notebook

## Overview

This Jupyter Notebook provides a complete end-to-end pipeline for fine-tuning a DETR-based object detection model (DETR `facebook/detr-resnet-50`) on the custom AUAIR aerial imagery dataset. It covers data loading, preprocessing, stratified splitting, training, evaluation, and testing within the notebook cells.

The project aims to detect objects like humans, cars, and trucks in aerial views, addressing challenges like small dataset size and class imbalance.

## Dataset

* **AUAIR Dataset:** Utilizes custom aerial images with JSON annotations.
* **Preprocessing:** Includes filtering invalid bounding boxes and data augmentation using `Albumentations`.
* **Splitting:** Implements a multi-label stratified split into Train, Validation, and Test sets directly within the notebook using `iterstrat`.

## Model & Training

* **Model:** Leverages pretrained models from Hugging Face Transformers via transfer learning.
* **Techniques:**
    * PyTorch framework.
    * AdamW optimizer with gradient clipping.
    * Evaluation using standard COCO metrics (`pycocotools`).
    * Experiment tracking integrated with Weights & Biases (`wandb`).

## Status

* The pipeline covers all steps from data loading to evaluation.
* Further training and hyperparameter tuning are necessary to achieve optimal detection performance.

## Usage

1.  **Environment Setup:**
    * Install the required dependencies listed in the `requirements.txt`. 
    * Log in to Weights & Biases in your terminal: `wandb login`
2.  **Data:**
    * Ensure the AUAIR dataset (images folder and `annotations.json`) is located at the path specified in the notebook's configuration cells (e.g., `./dataset/auair/`).
3.  **Configuration:**
    * Review the configuration cells near the beginning of the notebook. Adjust parameters like dataset paths, model checkpoint name, batch size, number of epochs, learning rate, wandb project details, etc., as needed.
4.  **Execution:**
    * Run the notebook cells sequentially from top to bottom. The notebook guides through data loading, splitting, model setup, training, and final evaluation/testing on the test set.

