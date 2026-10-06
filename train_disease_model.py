import os
import tensorflow as tf

from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau


# ============================================================
# CONFIGURATION
# ============================================================

DATASET_DIR = "dataset"

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32

INITIAL_EPOCHS = 10
FINE_TUNE_EPOCHS = 10

MODEL_FILE = "plant_disease_model.h5"


# ============================================================
# CHECK DATASET
# ============================================================

if not os.path.exists(DATASET_DIR):
    raise FileNotFoundError(
        "dataset folder was not found."
    )

class_folders = sorted([
    folder
    for folder in os.listdir(DATASET_DIR)
    if os.path.isdir(os.path.join(DATASET_DIR, folder))
])

if len(class_folders) == 0:
    raise FileNotFoundError(
        "No class folders found inside dataset."
    )

print("\nClasses found:")

for i, name in enumerate(class_folders):
    print(f"{i}: {name}")

print(f"\nTotal classes: {len(class_folders)}")


# ============================================================
# DATA AUGMENTATION
# ============================================================

datagen = ImageDataGenerator(
    rescale=1.0 / 255.0,

    rotation_range=20,
    width_shift_range=0.15,
    height_shift_range=0.15,
    zoom_range=0.15,
    shear_range=0.10,
    horizontal_flip=True,

    validation_split=0.20
)


# ============================================================
# TRAINING DATA
# ============================================================

train_data = datagen.flow_from_directory(
    DATASET_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="training",
    shuffle=True
)


# ============================================================
# VALIDATION DATA
# ============================================================

validation_data = datagen.flow_from_directory(
    DATASET_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation",
    shuffle=False
)


# ============================================================
# CLASS MAPPING
# ============================================================

print("\n========================================")
print("CLASS MAPPING")
print("========================================")

for name, index in train_data.class_indices.items():
    print(f"{index} -> {name}")

print("\nTraining images:", train_data.samples)
print("Validation images:", validation_data.samples)


# ============================================================
# MOBILE NET V2
# ============================================================

base_model = MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=(224, 224, 3)
)

# Freeze base model initially
base_model.trainable = False


# ============================================================
# BUILD MODEL
# ============================================================

model = models.Sequential([
    base_model,

    layers.GlobalAveragePooling2D(),

    layers.Dropout(0.30),

    layers.Dense(
        128,
        activation="relu"
    ),

    layers.Dropout(0.20),

    layers.Dense(
        len(class_folders),
        activation="softmax"
    )
])


# ============================================================
# COMPILE
# ============================================================

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.0001
    ),

    loss="categorical_crossentropy",

    metrics=["accuracy"]
)


# ============================================================
# CALLBACKS
# ============================================================

checkpoint = ModelCheckpoint(
    MODEL_FILE,
    monitor="val_accuracy",
    save_best_only=True,
    verbose=1
)

early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True
)

reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.3,
    patience=2,
    min_lr=0.000001,
    verbose=1
)


# ============================================================
# STAGE 1 - INITIAL TRAINING
# ============================================================

print("\n========================================")
print("STAGE 1: INITIAL TRAINING")
print("========================================")

history1 = model.fit(
    train_data,
    validation_data=validation_data,
    epochs=INITIAL_EPOCHS,
    callbacks=[
        checkpoint,
        early_stopping,
        reduce_lr
    ]
)


# ============================================================
# STAGE 2 - FINE TUNING
# ============================================================

print("\n========================================")
print("STAGE 2: FINE TUNING")
print("========================================")

base_model.trainable = True

# Freeze most MobileNetV2 layers
for layer in base_model.layers[:-30]:
    layer.trainable = False


# Recompile with very small learning rate
model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.00001
    ),

    loss="categorical_crossentropy",

    metrics=["accuracy"]
)


history2 = model.fit(
    train_data,
    validation_data=validation_data,
    epochs=FINE_TUNE_EPOCHS,
    callbacks=[
        checkpoint,
        early_stopping,
        reduce_lr
    ]
)


# ============================================================
# LOAD BEST MODEL
# ============================================================

model = tf.keras.models.load_model(MODEL_FILE)


# ============================================================
# FINAL EVALUATION
# ============================================================

loss, accuracy = model.evaluate(
    validation_data,
    verbose=1
)

print("\n========================================")
print("TRAINING COMPLETED")
print("========================================")

print(
    f"Validation Accuracy: {accuracy * 100:.2f}%"
)

print(
    f"Validation Loss: {loss:.4f}"
)

print(
    f"\nModel saved as: {MODEL_FILE}"
)

print(
    "\nRun your Streamlit application:"
)

print(
    "python -m streamlit run app.py"
)