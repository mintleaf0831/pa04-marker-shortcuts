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
# This script automatically reads models/seed_<seed>/ and writes data result/seed_<seed>/.
# Train that seed first. To evaluate an older run, select its SEED line and run only this file.

HEADERS = ("training_model", "no_marker", "matching_marker", "swapped_marker")


def evaluate(data_dir, model_dir, output_dir):
    models = {name: tf.keras.models.load_model(model_dir / f"model_{name}.keras", compile=False) for name in TRAIN_CONDITIONS}
    accuracy, macro_f1 = np.zeros((3, 3)), np.zeros((3, 3))
    for column, test in enumerate(TEST_CONDITIONS):
        images, labels = load_data(data_dir / "test" / test)
        if len(labels) != 4000 or not np.array_equal(np.bincount(labels), [1000] * 4):
            raise ValueError("Expected 1000 test images per class.")
        for row, name in enumerate(TRAIN_CONDITIONS):
            probs = models[name].predict(images, batch_size=128, verbose=0)
            if probs.shape != (4000, 4) or not np.isfinite(probs).all():
                raise ValueError("Expected finite predictions for four classes.")
            predicted = probs.argmax(axis=1)
            confusion = np.bincount(4 * labels + predicted, minlength=16).reshape(4, 4)
            accuracy[row, column] = np.trace(confusion) / confusion.sum() * 100
            macro_f1[row, column] = np.mean( 2 * confusion.diagonal() / (confusion.sum(axis=0) + confusion.sum(axis=1)))
            print(f"{name} / {test}: {accuracy[row, column]:.2f}%", flush=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    for filename, values in (("01_accuracy_percent.csv", accuracy), ("02_macro_f1.csv", macro_f1)):
        with (output_dir / filename).open("w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(HEADERS)
            decimals = 2 if filename.startswith("01") else 4
            writer.writerows([name, *[f"{v:.{decimals}f}" for v in values[row]]] for row, name in enumerate(TRAIN_CONDITIONS))
    return accuracy, macro_f1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "datasets")
    parser.add_argument("--models", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    if args.seed < 0:
        parser.error("Seed must be nonnegative.")
    model_dir = args.models or ROOT / "models" / f"seed_{args.seed}"
    output_dir = args.out or ROOT / "data result" / f"seed_{args.seed}"
    print(f"Evaluating seed {args.seed} from {model_dir}", flush=True)
    evaluate(args.data_dir, model_dir, output_dir)
