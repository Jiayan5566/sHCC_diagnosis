
This project involves the use of a 2D ResNet model for ultrasound image classification, including training and evaluation scripts. The model is based on the ResNet50 architecture, and the dataset is provided in the dt_DL.csv file.

Table of Contents
Installation
Usage
Training the Model
Evaluating the Model

Installation
Clone this repository:
git clone https://github.com/your-repo/project.git
cd project

Usage
Training the Model
The training script is 2d_ResNet_hjy.py, which reads ultrasound image paths and labels from the dt_DL.csv file. The script processes the images and feeds them into a modified ResNet50 model for binary classification.

Steps to Train the Model:
Prepare the dt_DL.csv file, ensuring it contains the following:
Column 3: Paths to the ultrasound image files.
Column 5: Corresponding binary classification labels (0 or 1).
Run the training script:
python 2d_ResNet_hjy.py

Evaluating the Model
Once the model has been trained, you can use the 2d_eva_new.py script to evaluate its performance and output prediction probabilities for each sample in the training and validation sets.
Run the evaluation script:
python 2d_eva_new.py
Script Details:
Data Loading: The script reads the image paths and labels from the dt_DL.csv file.
Evaluation: The script evaluates the model on both the training and validation sets and calculates:
Accuracy
AUC
Specificity (True Negatives / (True Negatives + False Positives))
Sensitivity (True Positives / (True Positives + False Negatives))
Output Files:
train_probabilities35.csv: This file contains the following columns for each sample in the training set:
DataName: Original file path of the sample.
Probability: The predicted probability of the positive class.
Label: The true label of the sample.
test_probabilities35.csv: This file contains the same columns for each sample in the validation set.
