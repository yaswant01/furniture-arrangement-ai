import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import backend as K

# Load dataset
X_train = pd.read_csv("room_data.csv").values
y_train = pd.read_csv("room_labels.csv").values

# Dynamically determine input size based on dataset
input_size = X_train.shape[1]  # Adjusts to dataset's actual shape

# Increase weight multiplier to prioritize furniture placements
WEIGHT_SCALE = 5.0  
weight_tensor = tf.constant(WEIGHT_SCALE, dtype=tf.float32)

# Define Weighted Binary Cross-Entropy loss function
def weighted_binary_crossentropy(y_true, y_pred):
    return K.mean(weight_tensor * K.binary_crossentropy(y_true, y_pred))

# Define a model that supports **variable room sizes**
model = keras.Sequential([
    keras.Input(shape=(input_size,)),  # Adjusts dynamically
    layers.Dense(256, activation="relu"),
    layers.BatchNormalization(),
    layers.Dropout(0.2),
    
    layers.Dense(128, activation="relu"),
    layers.BatchNormalization(),
    layers.Dropout(0.2),
    
    layers.Dense(64, activation="relu"),
    layers.Dense(input_size, activation="softmax")  # Matches input shape
])

# Compile Model
model.compile(optimizer="adam", loss=weighted_binary_crossentropy, metrics=["accuracy"])

# Train Model
model.fit(X_train, y_train, epochs=150, batch_size=32, verbose=1)

# Save Model
model.save("furniture_model.keras")  # Use new format

print("✅ Model Training Complete with Dynamic Input Handling!")
