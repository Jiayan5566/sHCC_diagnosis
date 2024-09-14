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
from keras.applications.inception_v3 import InceptionV3
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras.callbacks import ModelCheckpoint
from keras.metrics import TruePositives, TrueNegatives, FalsePositives, FalseNegatives

def process_scan(path):
    # Read scan
    if not isinstance(path, str):
        raise ValueError(f"Invalid path: {path}. Path should be a string.")
    volume = load_img(path,target_size=(224,224))
    volume = img_to_array(volume)
    return volume

# 读取 CSV 文件
df = pd.read_csv('dt_DL.csv')
file_path = df.iloc[:, 3].tolist()
data = np.array([process_scan(path) for path in file_path])
label = np.array(df.iloc[:, 5].tolist())
#label = to_categorical(label, num_classes=2)
#print(label[0])
#print(label.shape)

seed = 1
np.random.seed(seed)
tf.random.set_seed(seed)

x_train, x_val, y_train, y_val = train_test_split(data, label, test_size=0.4, random_state=seed)
print(
    "Number of samples in train and validation are %d and %d %d."
    % (x_train.shape[0], x_val.shape[0],data.shape[0])
)

num_classes = 2
batch=64

base_model = ResNet50(weights='imagenet', include_top=False)
for layer in base_model.layers:
    layer.trainable = False
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(1, activation='sigmoid')(x)
model = Model(inputs=base_model.input, outputs=predictions)

# Compile model.
initial_learning_rate = 0.0001
lr_schedule = keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate, decay_steps=100000, decay_rate=0.96, staircase=True
)
model.compile(
    loss="binary_crossentropy",
    optimizer=keras.optimizers.Adam(learning_rate=lr_schedule),
    metrics=['accuracy', tf.keras.metrics.AUC(name='auc'),
                           TruePositives(name='tp'),
                           TrueNegatives(name='tn'),
                           FalsePositives(name='fp'),
                           FalseNegatives(name='fn')],
    run_eagerly=False,
)
model.summary()
# Define callbacks.
#checkpoint_cb = keras.callbacks.ModelCheckpoint(
#    "3d_image_classification.keras", save_best_only=True)
#early_stopping_cb = keras.callbacks.EarlyStopping(monitor="val_acc", patience=15)
checkpoint = ModelCheckpoint(
    filepath='model-{epoch:02d}.hdf5',  # 指定保存路径和文件名格式
    save_weights_only=False,  # 设置为 True 仅保存权重，False 保存整个模型
    save_best_only=True,      # 仅保存验证集上表现最好的模型
    monitor='val_auc',       # 监控的指标
    mode='max',               # 'min' 表示指标越小越好
    verbose=1
)
# Train the model, doing validation at the end of each epoch
epochs = 100
model.fit(
    x_train,y_train,
    validation_data=(x_val,y_val),
    epochs=epochs,
    shuffle=True,batch_size=batch,
    verbose=1,
    callbacks=[checkpoint])
