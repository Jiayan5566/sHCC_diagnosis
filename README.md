
# Project Overview

This project involves the use of a 2D ResNet model for ultrasound image classification, including training and evaluation scripts. The model is based on the **ResNet50** architecture.

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
  - [Training the Model](#training-the-model)
  - [Evaluating the Model](#evaluating-the-model)

## Installation

Clone this repository:

```bash
git clone https://github.com/Jiayan5566/sHCC_diagnosis.git
cd project
```

## Usage

### Training the Model

The training script is `2d_ResNet_hjy.py`, which reads ultrasound image paths and labels from the `dt_DL.csv` file. The script processes the images and feeds them into a modified **ResNet50** model for binary classification.

#### Steps to Train the Model:

1. Prepare the `dt_DL.csv` file, ensuring it contains the following:
   - **Column 1**: Paths to the ultrasound image files.
   - **Column 2**: Corresponding binary classification labels (0 or 1).

2. Run the training script:

   ```bash
   python 2d_ResNet_hjy.py
   ```

### Evaluating the Model

Once the model has been trained, you can use the `2d_eva_new.py` script to evaluate its performance and output prediction probabilities for each sample in the training and validation sets.

#### Run the evaluation script:

```bash
python 2d_eva_new.py
```

#### Script Details:

- **Evaluation**: The script evaluates the model on both the training and validation sets and calculates:
  - **Accuracy**
  - **AUC**
  - **Specificity** (True Negatives / (True Negatives + False Positives))
  - **Sensitivity** (True Positives / (True Positives + False Negatives))


