import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Disables GPU detection completely
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"  # Disables deep learning acceleration

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

# 1. SETUP AND HYPERPARAMETERS
DATASET_PATH = "C:/Users/user/Desktop/Climbing Hold CNN/Clean_Classification_Dataset"
TRAIN_DIR = os.path.join(DATASET_PATH, 'train')
VAL_DIR = os.path.join(DATASET_PATH, 'valid')
TEST_DIR = os.path.join(DATASET_PATH, 'test')

IMG_SIZE = (224, 224)  # MobileNetV3 standard input size
BATCH_SIZE = 32        # Bumped up for smoother training gradients
NUM_CLASSES = 6        

# 2. LOAD DATASETS
print("Loading datasets...")
train_ds = keras.utils.image_dataset_from_directory(
    TRAIN_DIR, image_size=IMG_SIZE, batch_size=BATCH_SIZE, label_mode='categorical'
)
val_ds = keras.utils.image_dataset_from_directory(
    VAL_DIR, image_size=IMG_SIZE, batch_size=BATCH_SIZE, label_mode='categorical'
)
test_ds = keras.utils.image_dataset_from_directory(
    TEST_DIR, image_size=IMG_SIZE, batch_size=BATCH_SIZE, label_mode='categorical', shuffle=False
)

class_names = train_ds.class_names
print(f"Detected classes: {class_names}")

# Optimize data pipelines
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.shuffle(256).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)
test_ds = test_ds.prefetch(buffer_size=AUTOTUNE)

# 3. CONSTRUCT THE TRANSFER LEARNING ARCHITECTURE
print("Building Transfer Learning Model...")

# Load the pre-trained Google MobileNetV3 brain
base_model = keras.applications.MobileNetV3Large(
    input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3),
    include_top=False, # Drop the generic 1000-class head
    weights='imagenet'
)
base_model.trainable = False  # Freeze these features so we don't ruin them

model = models.Sequential([
    layers.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3)),
    
    # Data Augmentation: Vital for teaching the model variations in holds
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.15),
    layers.RandomContrast(0.15),
    
    # Pre-trained base
    base_model,
    
    # Custom Classification Head for climbing holds
    layers.GlobalAveragePooling2D(), 
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.4),            # Prevents overfitting
    layers.Dense(NUM_CLASSES, activation='softmax')
])

# 4. COMPILE AND TRAIN
# Crucial: Using a smaller learning rate (0.0005) so the model doesn't overcorrect
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.0005),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Early stopping prevents wasting time if the model stops improving
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True)

EPOCHS = 15  # Transfer learning converges MUCH faster (usually 5-15 epochs)
print("\nStarting Transfer Learning Training...")
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=[early_stop]
)

print("\nSaving optimized model...")
model.save("climbing_hold_model_optimized.keras")

# 5. EVALUATION
print("\n=== FINAL EVALUATION ===")
y_true = np.concatenate([y for x, y in test_ds], axis=0)
y_true_indices = np.argmax(y_true, axis=1)

predictions = model.predict(test_ds)
y_pred_indices = np.argmax(predictions, axis=1)

print(classification_report(y_true_indices, y_pred_indices, target_names=class_names))