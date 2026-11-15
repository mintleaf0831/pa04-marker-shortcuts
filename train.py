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

# Find this file's folder so the code can run from another working folder.
ROOT = Path(__file__).resolve().parent


def build_model():
    # The CNN learns small image patterns with two convolution layers.
    # Pooling makes the image maps smaller. Dense layers use these patterns to choose a class.
    # The input is a 32 x 32 gray image. Softmax gives four class probabilities.
    model = tf.keras.Sequential()
    model.add(tf.keras.Input(shape=(32, 32, 1)))

    # Find image patterns, then make the feature maps smaller.
    model.add(tf.keras.layers.Conv2D(16, 3, padding="same", activation="relu"))
    model.add(tf.keras.layers.MaxPooling2D())
    model.add(tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu"))
    model.add(tf.keras.layers.MaxPooling2D())

    # Combine the patterns and output four class probabilities.
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(32, activation="relu"))
    model.add(tf.keras.layers.Dense(4, activation="softmax"))

    # Cross-entropy measures prediction error. Adam updates the weights to reduce it.
    # The learning rate is 0.001. Accuracy tracks the share of correct predictions.
    model.compile(optimizer=tf.keras.optimizers.Adam(0.001), loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    return model


def train(data_dir, output_dir, seed=SEED, epochs=15):
    # Use repeatable TensorFlow operations on the same software and hardware.
    tf.config.experimental.enable_op_determinism()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Train a separate model for each marker condition with the same settings.
    for condition in TRAIN_CONDITIONS:
        # Load training images and check the class sizes.
        images, labels = load_data(data_dir / "train" / condition)
        class_counts = np.bincount(labels, minlength=4)
        if len(labels) != 2400 or list(class_counts) != [600, 600, 600, 600]:
            raise ValueError("Expected 600 training images per class.")

        # Clear the old model state and reset the seed before making a new model.
        # Each condition starts with new weights, using the same seed for a fair comparison.
        tf.keras.backend.clear_session()
        tf.keras.utils.set_random_seed(seed)
        print(f"Training {condition}, seed {seed}", flush=True)
        model = build_model()

        # One epoch uses the full training set once. Each batch has up to 64 images.
        # Shuffle the image order each epoch. Keras handles prediction, loss, and weight updates.
        # Use the final epoch model. This code does not use a validation set to select an epoch.
        model.fit(images, labels, epochs=epochs, batch_size=64, shuffle=True, verbose=2)

        # Save the trained model and its settings so it can be tested later.
        model.save(output_dir / f"model_{condition}.keras")
        settings = {"condition": condition, "seed": seed, "epochs": epochs, "batch_size": 64, "learning_rate": 0.001}
        # Record the software versions and the size of one input image.
        settings["tensorflow_version"] = tf.__version__
        settings["numpy_version"] = np.__version__
        settings["image_shape"] = list(images.shape[1:])
        settings_path = output_dir / f"settings_{condition}.json"
        settings_text = json.dumps(settings, indent=2)
        settings_path.write_text(settings_text + "\n", encoding="utf-8")


if __name__ == "__main__":
    # Command-line options can replace the defaults, including the selected seed.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "datasets")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--epochs", type=int, default=15)
    args = parser.parse_args()

    if args.seed < 0 or args.epochs < 1:
        parser.error("Seed must be nonnegative and epochs must be positive.")

    # Keep each seed in its own folder. Running that seed again replaces its files.
    output_dir = args.out or ROOT / "models" / f"seed_{args.seed}"
    print(f"Selected seed {args.seed}. Models will be saved in {output_dir}", flush=True)
    train(args.data_dir, output_dir, args.seed, args.epochs)

