"""
Ultra Target Cardiomegaly Training
Goal: push DenseNet121 cardiomegaly accuracy past 90% with staged fine-tuning,
advanced augmentation, and high-resolution inputs. Models + history files are
saved directly into the existing project model management structure.
"""

import json
import os
import random
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import tensorflow as tf
from tensorflow.keras import callbacks, layers, models, regularizers
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.applications.densenet import preprocess_input

# --------------------------------------------------------------------------------------
# Environment prep
# --------------------------------------------------------------------------------------
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
tf.get_logger().setLevel("ERROR")

SEED = 1337
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

physical_gpus = tf.config.list_physical_devices("GPU")
for gpu in physical_gpus:
    try:
        tf.config.experimental.set_memory_growth(gpu, True)
    except Exception:  # pragma: no cover
        pass

# --------------------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------------------
DATA_DIR = Path("Dataset/CHEST/cardiomelgy")
IMG_SIZE = (380, 380)
BATCH_SIZE = 16
VAL_SPLIT = 0.2
AUTOTUNE = tf.data.AUTOTUNE
NUM_CLASSES = 2
BASELINE_ACCURACY = 0.8036  # last advanced training best accuracy
TARGET_ACCURACY = 0.90

TRAINING_STAGES = [
    {
        "name": "warmup_head",
        "epochs": 6,
        "learning_rate": 8e-4,
        "trainable_back_layers": 60,  # only the classification head + last dense blocks
    },
    {
        "name": "mid_finetune",
        "epochs": 12,
        "learning_rate": 3e-4,
        "trainable_back_layers": 180,
    },
    {
        "name": "full_finetune",
        "epochs": 14,
        "learning_rate": 9e-5,
        "trainable_back_layers": None,  # unfreeze entire network
    },
]
TOTAL_EPOCHS = sum(stage["epochs"] for stage in TRAINING_STAGES)

print("=" * 96)
print("🚀 ULTRA DENSENET121 CARDIOMEGALY TRAINING — TARGET ≥ 90% ACCURACY")
print("=" * 96)
print(f"Data Directory        : {DATA_DIR}")
print(f"Image Size             : {IMG_SIZE}")
print(f"Batch Size             : {BATCH_SIZE}")
print(f"Training Stages        : {[stage['name'] for stage in TRAINING_STAGES]}")
print(f"Total Epoch Budget     : {TOTAL_EPOCHS}")
print(f"Baseline Accuracy      : {BASELINE_ACCURACY:.2%}")
print("=" * 96)

if not DATA_DIR.exists():
    raise FileNotFoundError(f"Dataset directory not found: {DATA_DIR}")

# --------------------------------------------------------------------------------------
# Dataset utilities
# --------------------------------------------------------------------------------------
def list_class_counts(root_dir: Path) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for child in sorted(root_dir.iterdir()):
        if child.is_dir():
            image_files = [fp for fp in child.iterdir() if fp.is_file()]
            counts[child.name] = len(image_files)
    if len(counts) != NUM_CLASSES:
        print(f"⚠️ Expected {NUM_CLASSES} classes, found {len(counts)}. Using detected classes.")
    return counts


def compute_class_weights(train_counts: Dict[str, int], class_names: List[str]) -> Dict[int, float]:
    total = sum(train_counts.values())
    weights: Dict[int, float] = {}
    for idx, name in enumerate(class_names):
        count = max(train_counts.get(name, 1), 1)
        weights[idx] = total / (len(class_names) * count)
    return weights


print("📂 Indexing dataset...")
class_names = sorted([d.name for d in DATA_DIR.iterdir() if d.is_dir()])
if len(class_names) != NUM_CLASSES:
    NUM_CLASSES = len(class_names)
print(f"✅ Classes detected: {class_names}")

raw_class_counts = list_class_counts(DATA_DIR)
train_class_counts = {cls: int(count * (1.0 - VAL_SPLIT)) for cls, count in raw_class_counts.items()}
class_weights = compute_class_weights(train_class_counts, class_names)
print(f"📊 Estimated class weights: {class_weights}\n")

print("📦 Building tf.data pipelines...")

train_ds_raw = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    labels="inferred",
    label_mode="int",
    validation_split=VAL_SPLIT,
    subset="training",
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
)
val_ds_raw = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    labels="inferred",
    label_mode="int",
    validation_split=VAL_SPLIT,
    subset="validation",
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
)

total_sample_count = sum(raw_class_counts.values())
train_sample_count = int(total_sample_count * (1.0 - VAL_SPLIT))
val_sample_count = int(total_sample_count - train_sample_count)
print(f"✅ Training samples (est.): {train_sample_count}")
print(f"✅ Validation samples (est.): {val_sample_count}")

augmentation = tf.keras.Sequential(
    [
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.08),
        layers.RandomTranslation(0.08, 0.08),
        layers.RandomZoom(0.1),
        layers.RandomContrast(0.15),
    ],
    name="strong_augmentation",
)


def preprocess_batch(images, labels, training: bool):
    images = tf.cast(images, tf.float32)
    if training:
        images = augmentation(images, training=True)
    images = preprocess_input(images)
    labels = tf.one_hot(labels, depth=NUM_CLASSES)
    return images, labels


train_ds = train_ds_raw.map(lambda x, y: preprocess_batch(x, y, True), num_parallel_calls=AUTOTUNE)
train_ds = train_ds.shuffle(1024, seed=SEED).prefetch(AUTOTUNE)

val_ds = val_ds_raw.map(lambda x, y: preprocess_batch(x, y, False), num_parallel_calls=AUTOTUNE)
val_ds = val_ds.cache().prefetch(AUTOTUNE)

# --------------------------------------------------------------------------------------
# Model setup
# --------------------------------------------------------------------------------------
print("🧠 Building DenseNet121 backbone...")
base_model = DenseNet121(include_top=False, weights="imagenet", input_shape=IMG_SIZE + (3,))

inputs = layers.Input(shape=IMG_SIZE + (3,), name="input_image")
x = inputs
x = base_model(x, training=False)
x = layers.GlobalAveragePooling2D(name="avg_pool")(x)
x = layers.BatchNormalization()(x)
x = layers.Dense(640, activation="relu", kernel_regularizer=regularizers.l2(1e-4))(x)
x = layers.Dropout(0.55)(x)
x = layers.BatchNormalization()(x)
x = layers.Dense(320, activation="relu", kernel_regularizer=regularizers.l2(1e-4))(x)
x = layers.Dropout(0.45)(x)
x = layers.Dense(160, activation="relu", kernel_regularizer=regularizers.l2(5e-5))(x)
x = layers.Dropout(0.35)(x)
outputs = layers.Dense(NUM_CLASSES, activation="softmax", dtype="float32", name="predictions")(x)

model = models.Model(inputs=inputs, outputs=outputs, name="densenet121_cardiomegaly_ultra")

model.summary()

model_dir = Path("models/cardiomegaly")
model_dir.mkdir(parents=True, exist_ok=True)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
best_model_path = model_dir / f"DenseNet121_ultra90_{timestamp}.h5"
final_model_path = Path("models") / "DenseNet121_cardiomegaly_ultra.h5"

class TargetAccuracyCallback(callbacks.Callback):
    def __init__(self, target: float):
        super().__init__()
        self.target = target
        self.best = 0.0

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        val_accuracy = logs.get("val_accuracy")
        if val_accuracy is None:
            return
        if val_accuracy > self.best:
            self.best = val_accuracy
            gap = (self.target - val_accuracy) * 100
            print(
                f"\n🎯 New best val accuracy: {val_accuracy:.2%} | Gap to target: {max(gap, 0):.2f}%"
            )
        if val_accuracy >= self.target:
            print("\n🎉 Target accuracy reached — stopping early!")
            self.model.stop_training = True


def set_trainable_layers(model: tf.keras.Model, back_layers: Optional[int]) -> int:
    total_layers = len(base_model.layers)
    if back_layers is None or back_layers >= total_layers:
        for layer in base_model.layers:
            layer.trainable = True
        return total_layers
    cutoff = total_layers - back_layers
    for idx, layer in enumerate(base_model.layers):
        layer.trainable = idx >= cutoff
    return back_layers


def compile_model(lr: float):
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-7)
    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.04),
        metrics=[
            "accuracy",
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.AUC(name="auc"),
        ],
    )


training_callbacks = [
    TargetAccuracyCallback(TARGET_ACCURACY),
    callbacks.ModelCheckpoint(
        filepath=str(best_model_path),
        monitor="val_accuracy",
        mode="max",
        save_best_only=True,
        save_weights_only=False,
        verbose=1,
    ),
    callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.35,
        patience=4,
        min_lr=1e-7,
        verbose=1,
    ),
    callbacks.EarlyStopping(
        monitor="val_accuracy",
        patience=9,
        mode="max",
        restore_best_weights=True,
        verbose=1,
    ),
]

full_history: Dict[str, List[float]] = {}
current_epoch = 0

for stage in TRAINING_STAGES:
    print("\n" + "-" * 80)
    print(f"🌀 Stage: {stage['name']} | epochs: {stage['epochs']} | lr: {stage['learning_rate']}")
    trainable = set_trainable_layers(model, stage["trainable_back_layers"])
    print(f"🔓 Trainable DenseNet layers this stage: {trainable}")
    compile_model(stage["learning_rate"])

    stage_history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=current_epoch + stage["epochs"],
        initial_epoch=current_epoch,
        class_weight=class_weights,
        callbacks=training_callbacks,
        verbose=1,
    )

    for key, values in stage_history.history.items():
        full_history.setdefault(key, []).extend(values)

    current_epoch += stage["epochs"]
    if stage_history.history.get("val_accuracy") and max(stage_history.history["val_accuracy"]) >= TARGET_ACCURACY:
        break

print("\n" + "=" * 96)
print("📊 TRAINING COMPLETE — GENERATING SUMMARY")
print("=" * 96)

if best_model_path.exists():
    model = tf.keras.models.load_model(str(best_model_path))
    print(f"✅ Loaded best checkpoint: {best_model_path}")
else:
    print("⚠️ Best checkpoint missing; saving current model as best.")
    model.save(str(best_model_path))

model.save(str(final_model_path))
print(f"💾 Active project model updated: {final_model_path}")

best_val_accuracy = max(full_history.get("val_accuracy", [0.0]))
final_val_accuracy = full_history.get("val_accuracy", [0.0])[-1]
best_train_accuracy = max(full_history.get("accuracy", [0.0]))

print(f"Baseline Accuracy : {BASELINE_ACCURACY:.2%}")
print(f"Best Validation   : {best_val_accuracy:.2%}")
print(f"Final Validation  : {final_val_accuracy:.2%}")
print(f"Best Training     : {best_train_accuracy:.2%}")
print(f"Target Achieved   : {'YES' if best_val_accuracy >= TARGET_ACCURACY else 'NO'}")

history_payload = {
    "dataset": str(DATA_DIR),
    "classes": class_names,
    "training_samples": int(train_sample_count),
    "validation_samples": int(val_sample_count),
    "baseline_accuracy": float(BASELINE_ACCURACY),
    "best_val_accuracy": float(best_val_accuracy),
    "final_val_accuracy": float(final_val_accuracy),
    "best_train_accuracy": float(best_train_accuracy),
    "improvement": float(best_val_accuracy - BASELINE_ACCURACY),
    "epochs_trained": len(full_history.get("val_accuracy", [])),
    "target_achieved": bool(best_val_accuracy >= TARGET_ACCURACY),
    "timestamp": timestamp,
    "config": {
        "image_size": list(IMG_SIZE),
        "batch_size": BATCH_SIZE,
        "validation_split": VAL_SPLIT,
        "stages": TRAINING_STAGES,
        "trainable_params": int(np.sum([np.prod(v.shape) for v in model.trainable_variables])),
        "total_params": int(model.count_params()),
    },
    "history": {key: [float(val) for val in values] for key, values in full_history.items()},
}

history_path = model_dir / f"history_ultra90_{timestamp}.json"
with open(history_path, "w", encoding="utf-8") as fp:
    json.dump(history_payload, fp, indent=2)

print(f"📄 Training history saved: {history_path}")
print("=" * 96)
print("✅ ULTRA TRAINING SCRIPT FINISHED — READY FOR INTEGRATION")
print("=" * 96)
