import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Disables GPU detection completely
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"  # Disables deep learning acceleration c


import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

# Scikit-learn for evaluation metrics
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

# ----------------------------------------------------
# 1. SETUP AND HYPERPARAMETERS
# ----------------------------------------------------
# Replace this path with where your extracted dataset lives
DATASET_PATH = "C:/Users/user/Desktop/Climbing Hold CNN/Clean_Classification_Dataset"

TRAIN_DIR = os.path.join(DATASET_PATH, 'train')
VAL_DIR = os.path.join(DATASET_PATH, 'valid')
TEST_DIR = os.path.join(DATASET_PATH, 'test')

IMG_SIZE = (256, 256)  # Target dimensions to resize images
BATCH_SIZE = 16
NUM_CLASSES = 6        # Jug, Crimp, Pinch, Slope, Pocket, Volume

# ----------------------------------------------------
# 2. LOAD DATASETS USING KERAS UTILITIES
# ----------------------------------------------------
print("Loading datasets...")

train_ds = keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='categorical' # Converts labels to 1-hot encodings (6 classes)
)

val_ds = keras.utils.image_dataset_from_directory(
    VAL_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='categorical'
)

# Crucial: Keep shuffle=False for the test dataset so the labels line up 
# perfectly when evaluating performance via Scikit-Learn later.
test_ds = keras.utils.image_dataset_from_directory(
    TEST_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='categorical',
    shuffle=False 
)

# Extract class names to look up what indices map to what hold types
class_names = train_ds.class_names
print(f"Detected classes: {class_names}")

# Optimize data pipelines for memory buffering
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

# ----------------------------------------------------
# 3. CONSTRUCT THE CNN ARCHITECTURE
# ----------------------------------------------------
model = models.Sequential([
    # Input Layer and Data Augmentation / Preprocessing
    layers.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3)),
    
    # Rescale pixel values from [0, 255] to [0, 1]
    layers.Rescaling(1./255),
    
    # Data Augmentation (Helps generalize across synthetic & real gym photos)
    # layers.RandomFlip("horizontal_and_vertical"),
    # layers.RandomRotation(0.2),
    
    # 1st Convolution Block
    layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    
    # 2nd Convolution Block
    layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    
    # 3rd Convolution Block
    layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    
    # Classification Head
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5), # Prevents overfitting
    layers.Dense(NUM_CLASSES, activation='softmax') # Multi-class probability array
])

model.summary()

# ----------------------------------------------------
# 4. COMPILE AND TRAIN
# ----------------------------------------------------
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

EPOCHS = 20
print("\nStarting CNN Training...")
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS
)


print("\nSaving trained model...")
# Saves the entire model architecture, weights, and optimizer state
model.save("climbing_hold_model.keras") 
print("Model saved successfully as 'climbing_hold_model.keras'!")


# ----------------------------------------------------
# 5. SCIKIT-LEARN EVALUATION
# ----------------------------------------------------
print("\nEvaluating Model with Scikit-Learn...")

# Extract true labels from our un-shuffled test dataset
y_true = np.concatenate([y for x, y in test_ds], axis=0)
y_true_indices = np.argmax(y_true, axis=1) # Convert one-hot matrix back to integers

# Generate model predictions on the test set
predictions = model.predict(test_ds)
y_pred_indices = np.argmax(predictions, axis=1) # Target highest class probability

# Print Scikit-Learn's text-based Classification Report
print("\n=== CLASSIFICATION REPORT ===")
print(classification_report(y_true_indices, y_pred_indices, target_names=class_names))

# Generate and plot Confusion Matrix using Scikit-Learn's utilities
cm = confusion_matrix(y_true_indices, y_pred_indices)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)

fig, ax = plt.subplots(figsize=(8, 8))
disp.plot(cmap=plt.cm.Blues, ax=ax, xticks_rotation=45)
plt.title("Climbing Hold Classification - Confusion Matrix")
plt.show()