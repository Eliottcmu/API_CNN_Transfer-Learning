# CNN Transfer Learning – Fine-Tuning Pre-trained Models on Custom Data

## Overview

This project demonstrates a complete deep learning pipeline for fine-tuning a pre-trained convolutional neural network on a custom image dataset. The workflow covers automated data acquisition via Pixabay API, dataset structuring, preprocessing, data augmentation, transfer learning, and model evaluation.

The objective is to classify images into three categories:

* Strawberry
* Banana
* Pineapple

The implementation is provided as a Jupyter Notebook optimised for execution in Google Colab with GPU

---

## Key Features

* Automated dataset creation using Pixabay API
* Image validation and exploratory data analysis
* Train / validation / test split (70 / 15 / 15)
* Data augmentation pipeline using Keras ImageDataGenerator
* Transfer learning with ImageNet pre-trained models
* Fine-tuning of top layers
* Performance evaluation and visualisation
* Prediction on unseen images

---

## Technical Pipeline

### 1. Data Acquisition

* Images are fetched automatically using Pixabay API.
* Each category is downloaded and stored locally.

### 2. Data Validation & Exploration

* Image integrity checks
* Visual sampling
* Analysis of resolution and formats

### 3. Preprocessing & Augmentation

* Resizing to ImageNet-compatible dimensions
* Normalisation
* Random transformations:

  * Rotation
  * Zoom
  * Horizontal shift
  * Shear

### 4. Model Architecture

* Pre-trained CNN backbone (ImageNet weights)
* Frozen base layers for feature extraction
* Custom dense classifier head
* Optional unfreezing for fine-tuning

### 5. Training Strategy

* Categorical cross-entropy loss
* Adam optimizer
* Early stopping and model checkpointing

### 6. Evaluation

* Accuracy and loss curves
* Confusion matrix
* Inference on new images

---

## Requirements

Recommended environment: Google Colab

Dependencies:

* Python 3.x
* TensorFlow / Keras
* NumPy
* Matplotlib
* Seaborn
* Pillow
* Requests
* scikit-learn

Install locally if needed:

```
pip install tensorflow numpy matplotlib seaborn pillow requests scikit-learn
```

---

## Usage

### 1. Clone the repository

```
git clone https://github.com/Eliottcmu/API_CNN_Transfer-Learning.git
cd API_CNN_Transfer-Learning
```

### 2. Open notebook

Run in Google Colab for GPU support:

```
Eliott_Camou_Chapter17_Fine_Tuning_Transfer_Learning_Student_File.ipynb
```

### 3. Configure the dataset

You can edit categories and image count:

```python
CATEGORIES = ['strawberry', 'banana', 'pineapple']
IMAGES_PER_CATEGORY = 100
```
Add your api key and edit the query : 

```python
params = {
      'key' : PIXABAY_API_KEY,
      'q': query,
      'image_type': 'photo',
      'per_page': max(per_page, 200),
      'safesearch': 'true',
      'order': 'popular',
      'category': 'food', #change category for exemple
      }
```

### 4. Train and evaluate

Execute cells sequentially to:

* Download data
* Build dataset
* Train model
* Visualize performance

---

## Results

The model successfully learns high-level visual patterns using transfer learning, achieving stable convergence and improved generalisation despite limited dataset size.

Performance depends on:

* Number of training images
* Data augmentation intensity
* Fine-tuning depth

---

## Improvements & Extensions

* Replace Pixabay with custom dataset
* Add more classes
* Hyperparameter optimisation

---

## Author

Eliott Camou
