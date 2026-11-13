"""Save two small comparison tables for the three trained models."""

import argparse
import csv
import os
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import numpy as np
import tensorflow as tf
from build_datasets import TEST, TRAIN, load_images

HEADERS = ("training_model", "no_marker", "matching_marker", "swapped_marker")


def evaluate(data_dir, model_dir, output_dir):
    models = {name: tf.keras.models.load_model(model_dir / f"model_{name}.keras",
                                               compile=False) for name in TRAIN}
    accuracy, f1_scores = np.zeros((3, 3)), np.zeros((3, 3))
    for column, test in enumerate(TEST):
        images, labels = load_images(data_dir / "test" / test)
        if len(labels) != 4000 or not np.array_equal(np.bincount(labels), [1000] * 4):
            raise ValueError("Expected 1000 test images per class.")
        for row, name in enumerate(TRAIN):
            scores = models[name].predict(images, batch_size=128, verbose=0)
            if scores.shape != (4000, 4) or not np.isfinite(scores).all():
                raise ValueError("Expected finite predictions for four classes.")
            predicted = scores.argmax(axis=1)
            matrix = np.bincount(4 * labels + predicted, minlength=16).reshape(4, 4)
            accuracy[row, column] = np.trace(matrix) / matrix.sum() * 100
            f1_scores[row, column] = np.mean(
                2 * matrix.diagonal() / (matrix.sum(axis=0) + matrix.sum(axis=1)))
            print(f"{name} / {test}: {accuracy[row, column]:.2f}%", flush=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    for filename, values in (("01_accuracy_percent.csv", accuracy),
                             ("02_macro_f1.csv", f1_scores)):
        with (output_dir / filename).open("w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(HEADERS)
            decimals = 2 if filename.startswith("01") else 4
            writer.writerows([name, *[f"{v:.{decimals}f}" for v in values[row]]]
                             for row, name in enumerate(TRAIN))
    return accuracy, f1_scores


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("datasets"))
    parser.add_argument("--models", type=Path, default=Path("models"))
    parser.add_argument("--out", type=Path, default=Path("data result/evaluation"))
    args = parser.parse_args()
    evaluate(args.data_dir, args.models, args.out)
