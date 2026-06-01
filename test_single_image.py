import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Keep using CPU

import numpy as np
import tensorflow as tf
from tensorflow import keras

# 1. DEFINE YOUR SETTINGS (Must match your training script exactly)
MODEL_PATH = "climbing_hold_model_optimized.keras"
# Replace this with the path to the new image you want to test
IMAGE_PATH = "C:/Users/user/Desktop/Climbing Hold CNN/Test images/penis.png" 

IMG_SIZE = (224, 224) # Use whatever size you ended up training with (e.g., 128, 128)
class_names = ['Crimp', 'Jug', 'Pinch', 'Pocket', 'Slope', 'Volume'] # Ensure order matches training

# 2. LOAD THE TRAINED MODEL
print("Loading model...")
model = keras.models.load_model(MODEL_PATH)

# 3. LOAD AND PREPROCESS THE NEW IMAGE
print(f"Processing image: {IMAGE_PATH}")
# Load image using Keras utility
img = keras.utils.load_img(IMAGE_PATH, target_size=IMG_SIZE)

# Convert image to a numpy array
img_array = keras.utils.img_to_array(img)

# Models don't look at single images; they look at batches.
# We must expand dimensions to turn (height, width, channels) into (1, height, width, channels)
img_array = tf.expand_dims(img_array, 0) 

# 4. RUN PREDICTION
print("Running prediction...")
predictions = model.predict(img_array)

# 5. INTERPRET THE RESULTS
# predictions[0] contains an array of probabilities for each class (e.g., [0.01, 0.92, 0.03, ...])
predicted_class_idx = np.argmax(predictions[0])
predicted_class_name = class_names[predicted_class_idx]
confidence = predictions[0][predicted_class_idx] * 100

print("\n==============================")
print(f"Predicted Hold Type: {predicted_class_name}")
print(f"Confidence: {confidence:.2f}%")
print("==============================")

# Optional: Print out the breakdown for all classes
print("\nFull confidence breakdown:")
for name, prob in zip(class_names, predictions[0]):
    print(f"  {name}: {prob * 100:.2f}%")