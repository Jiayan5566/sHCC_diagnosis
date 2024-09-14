from keras.models import load_model
import pandas as pd
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import numpy as np
import keras
from keras.preprocessing.image import load_img, img_to_array
from keras.utils import to_categorical
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.applications.resnet50 import ResNet50
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras.callbacks import ModelCheckpoint
from keras.metrics import TruePositives, TrueNegatives, FalsePositives, FalseNegatives
from sklearn.metrics import f1_score, jaccard_score, accuracy_score,\
    classification_report, precision_score, recall_score, roc_curve, auc

def process_scan(path):
    # Read scan
    if not isinstance(path, str):
        raise ValueError(f"Invalid path: {path}. Path should be a string.")
    volume = load_img(path, target_size=(224, 224))
    volume = img_to_array(volume)
    return volume

# 读取 CSV 文件
df = pd.read_csv('dt_DL.csv')
file_path = df.iloc[:, 3].tolist()
data = np.array([process_scan(path) for path in file_path])
label = np.array(df.iloc[:, 5].tolist())

seed = 1
np.random.seed(seed)
tf.random.set_seed(seed)

x_train, x_val, y_train, y_val, train_file_path, val_file_path = train_test_split(
    data, label, file_path, test_size=0.4, random_state=seed
)

print(
    "Number of samples in train and validation are %d and %d %d."
    % (x_train.shape[0], x_val.shape[0], data.shape[0])
)
model = load_model('model-3-5.hdf5')

# Train set evaluation
_, accuracy, auc, tp, tn, fp, fn = model.evaluate(x_train, y_train)
specificity = tn / (tn + fp)
sensitivity = tp / (tp + fn)
print("Train set")
print("Accuracy in train:", accuracy)
print("Auc in train:", auc)
print("specificity in train:", specificity)
print("sensitivity in train:", sensitivity)

# Test set evaluation
_, accuracy, auc, tp, tn, fp, fn = model.evaluate(x_val, y_val, batch_size=1)
specificity = tn / (tn + fp)
sensitivity = tp / (tp + fn)
print("Test set")
print("Accuracy:", accuracy)
print("Auc:", auc)
print("specificity:", specificity)
print("sensitivity:", sensitivity)

# Output probability for train set
results = []
train_predictions = model.predict(x_train)
for i in range(len(train_predictions)):
    data_name = train_file_path[i]  # Use the original file path as the data name
    probability = train_predictions[i]  # Get the probability of the positive class (label 1)
    label = y_train[i]
    results.append([data_name, probability, label])  # Add data name and probability to the list

# Create a pandas DataFrame from the results list
results_df = pd.DataFrame(results, columns=["DataName", "Probability", "Label"])

# Save the DataFrame to a CSV file
results_df.to_csv("train_probabilities35.csv", index=False)

# Output probability for test set
results = []
test_predictions = model.predict(x_val)
for i in range(len(test_predictions)):
    data_name = val_file_path[i]  # Use the original file path as the data name
    probability1 = test_predictions[i]  # Get the probability of the positive class (label 1)
    label1 = y_val[i]
    results.append([data_name, probability1, label1])  # Add data name and probability to the list

# Create a pandas DataFrame from the results list
results_df = pd.DataFrame(results, columns=["DataName", "Probability", "Label"])

# Save the DataFrame to a CSV file
results_df.to_csv("test_probabilities35.csv", index=False)
