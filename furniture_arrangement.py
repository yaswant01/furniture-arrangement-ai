import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras import backend as K

# Define the custom loss function again so it's available when loading the model
def weighted_binary_crossentropy(y_true, y_pred):
    weight_tensor = tf.constant(5.0, dtype=tf.float32)  # Must match training loss
    return K.mean(weight_tensor * K.binary_crossentropy(y_true, y_pred))

# Load the model with custom objects
model = load_model("furniture_model.keras", custom_objects={"weighted_binary_crossentropy": weighted_binary_crossentropy})

st.title("Final AI-Based Furniture Arrangement")
st.write("This AI model suggests optimal furniture placements while avoiding obstacles.")

# **1️⃣ User Inputs: Define Room Size**
room_width = st.number_input("Room Width", min_value=5, max_value=20, value=10)
room_height = st.number_input("Room Height", min_value=5, max_value=20, value=10)

# **2️⃣ User Inputs: Number of Furniture & Obstacles**
num_furniture = st.slider("Number of Furniture Pieces", 1, 5, 3)
num_obstacles = st.slider("Number of Obstacles", 0, 5, 2)

# **3️⃣ User-Defined Obstacles**
obstacle_positions = set()
for i in range(num_obstacles):
    x = st.number_input(f"Obstacle {i+1} X", min_value=0, max_value=room_width-1, value=0)
    y = st.number_input(f"Obstacle {i+1} Y", min_value=0, max_value=room_height-1, value=0)
    obstacle_positions.add((x, y))

# **4️⃣ Generate Room Layout**
def generate_room_with_constraints():
    room = np.zeros((room_height, room_width))  # Adjust dynamically

    # Place obstacles as per user input
    for x, y in obstacle_positions:
        room[y, x] = -1  

    return room.flatten().reshape(1, -1)  # Reshape based on user size

room_input = generate_room_with_constraints()

# **5️⃣ Resize Input Dynamically to Match Model**
expected_input_size = model.input_shape[1]  # Get model's expected input size
room_input_resized = np.resize(room_input, (1, expected_input_size))  # Resize to match model

# **6️⃣ Predict Furniture Placement**
predictions = model.predict(room_input_resized).reshape(room_height, room_width)

# **7️⃣ Normalize Predictions**
max_value = np.max(predictions)
if max_value > 0:  
    predictions = predictions / max_value  

# **8️⃣ Display Optimized Furniture Placement**
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(predictions, cmap="coolwarm", linewidths=0.5, annot=True, fmt=".2f", ax=ax)
plt.title("Optimized AI Furniture Placement")
st.pyplot(fig)
