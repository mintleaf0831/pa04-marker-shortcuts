"""Train the same CNN with no markers, fixed markers and random markers."""

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import numpy as np
import tensorflow as tf
from data import TRAIN_CONDITIONS, load_data

# MANUAL SEED SELECTION
# Keep exactly ONE SEED line active (without #).
# Example: comment SEED = 42 and uncomment SEED = 43 to use seed 43.
# Save this file, run train.py, then run evaluate.py.
# Evaluation reads this setting automatically. No second seed edit is needed.
# The seed changes model initialization and shuffling, not the dataset.
SEED = 42
# SEED = 43
# SEED = 44
# SEED = 45
# SEED = 46

ROOT = Path(__file__).resolve().parent



def build_model():
    model = tf.keras.Sequential([ tf.keras.Input(shape=(32, 32, 1)), tf.keras.layers.Conv2D(16, 3, padding="same", activation="relu"), tf.keras.layers.MaxPooling2D(), tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu"), tf.keras.layers.MaxPooling2D(), tf.keras.layers.Flatten(), tf.keras.layers.Dense(32, activation="relu"), tf.keras.layers.Dense(4, activation="softmax"), ])
    model.compile(optimizer=tf.keras.optimizers.Adam(0.001), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def train(data_dir, output_dir, seed=SEED, epochs=15):
    tf.config.experimental.enable_op_determinism()
    output_dir.mkdir(parents=True, exist_ok=True)
    for condition in TRAIN_CONDITIONS:
        images, labels = load_data(data_dir / "train" / condition)
        if len(labels) != 2400 or not np.array_equal(np.bincount(labels), [600] * 4):
            raise ValueError("Expected 600 training images per class.")
        tf.keras.backend.clear_session()
        tf.keras.utils.set_random_seed(seed)
        print(f"Training {condition}, seed {seed}", flush=True)
        model = build_model()
        model.fit(images, labels, epochs=epochs, batch_size=64, shuffle=True, verbose=2)
        model.save(output_dir / f"model_{condition}.keras")
        settings = {"condition": condition, "seed": seed, "epochs": epochs, "batch_size": 64, "learning_rate": 0.001, "tensorflow": tf.__version__}
        (output_dir / f"settings_{condition}.json").write_text( json.dumps(settings, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "datasets")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--epochs", type=int, default=15)
    args = parser.parse_args()
    if args.seed < 0 or args.epochs < 1:
        parser.error("Seed must be nonnegative and epochs must be positive.")
    output_dir = args.out or ROOT / "models" / f"seed_{args.seed}"
    print(f"Selected seed {args.seed}. Models will be saved in {output_dir}", flush=True)
    train(args.data_dir, output_dir, args.seed, args.epochs)

