"""Save two small comparison tables for the three trained models."""

import argparse
import csv
import os
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import numpy as np
import tensorflow as tf
from data import TEST_CONDITIONS, TRAIN_CONDITIONS, load_data
from train import ROOT, SEED

# Choose the seed in train.py and save that file before evaluation.
# This script automatically reads models/seed_<seed>/ and writes results/seed_<seed>/.
# Train that seed first. To evaluate an older run, select its SEED line and run only this file.

HEADERS = ("training_model", "no_marker", "matching_marker", "swapped_marker")


def evaluate(data_dir, model_dir, output_dir):
    # Load the three saved models for prediction. No training takes place here.
    models = {}
    for name in TRAIN_CONDITIONS:
        model_path = model_dir / f"model_{name}.keras"
        models[name] = tf.keras.models.load_model(model_path, compile=False)

    # Rows represent training conditions. Columns represent test conditions.
    accuracy, macro_f1 = np.zeros((3, 3)), np.zeros((3, 3))

    # Test every model on each of the three versions of the same test images.
    for column, test in enumerate(TEST_CONDITIONS):
        # Load the test images and check the class sizes.
        images, labels = load_data(data_dir / "test" / test)
        class_counts = np.bincount(labels, minlength=4)
        if len(labels) != 4000 or list(class_counts) != [1000, 1000, 1000, 1000]:
            raise ValueError("Expected 1000 test images per class.")

        for row, name in enumerate(TRAIN_CONDITIONS):
            # Predict class probabilities, check them, and choose the most likely class.
            probs = models[name].predict(images, batch_size=128, verbose=0)
            if probs.shape != (4000, 4) or not np.isfinite(probs).all():
                raise ValueError("Expected finite predictions for four classes.")
            predicted = probs.argmax(axis=1)

            # Count true classes in rows and predicted classes in columns.
            # The diagonal counts correct answers. Use these counts to measure performance.
            # Accuracy is the percent of correct answers. Macro-F1 gives each class equal weight.
            # Each class F1 is twice its correct count divided by its true count plus predicted count.
            confusion = np.zeros((4, 4), dtype=int)
            for actual, guess in zip(labels, predicted):
                confusion[actual, guess] += 1

            correct_count = np.sum(predicted == labels)
            accuracy[row, column] = correct_count / len(labels) * 100

            # Calculate one F1 score per class, then take their mean.
            class_f1 = []
            for label in range(4):
                correct = confusion[label, label]
                actual_count = confusion[label, :].sum()
                predicted_count = confusion[:, label].sum()
                f1 = 2 * correct / (actual_count + predicted_count)
                class_f1.append(f1)
            macro_f1[row, column] = np.mean(class_f1)
            print(f"{name} / {test}: {accuracy[row, column]:.2f}%", flush=True)

    # Save one small table per metric. Accuracy uses percent; Macro-F1 uses 0-1.
    output_dir.mkdir(parents=True, exist_ok=True)
    for filename, values in (("01_accuracy_percent.csv", accuracy), ("02_macro_f1.csv", macro_f1)):
        with (output_dir / filename).open("w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(HEADERS)
            decimals = 2 if filename.startswith("01") else 4
            for row, name in enumerate(TRAIN_CONDITIONS):
                cells = [name]
                for value in values[row]:
                    cells.append(f"{value:.{decimals}f}")
                writer.writerow(cells)

    return accuracy, macro_f1


if __name__ == "__main__":
    # Use the seed from train.py unless a command-line option gives another value.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "datasets")
    parser.add_argument("--models", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    if args.seed < 0:
        parser.error("Seed must be nonnegative.")

    # Read the selected seed's models and save its results in a separate folder.
    model_dir = args.models or ROOT / "models" / f"seed_{args.seed}"
    output_dir = args.out or ROOT / "results" / f"seed_{args.seed}"
    print(f"Evaluating seed {args.seed} from {model_dir}", flush=True)
    evaluate(args.data_dir, model_dir, output_dir)
