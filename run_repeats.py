"""Train and evaluate five seeds, then report means and sample standard deviations."""

import argparse
import csv
from pathlib import Path

import numpy as np
from build_datasets import TEST, TRAIN
from evaluate_pa04 import evaluate
from train_pa04 import train


def repeat(data_dir, output_dir, seeds):
    results = []
    for seed in seeds:
        folder = output_dir / f"seed_{seed}"
        train(data_dir, folder / "models", seed)
        results.append(evaluate(data_dir, folder / "models", folder / "evaluation"))
    accuracy = np.stack([result[0] for result in results])
    f1 = np.stack([result[1] for result in results])
    with (output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["model", "test", "accuracy_mean_percent", "accuracy_sd_pp",
                         "macro_f1_mean", "macro_f1_sd"])
        for row, model in enumerate(TRAIN):
            for column, test in enumerate(TEST):
                writer.writerow([model, test, f"{accuracy[:, row, column].mean():.3f}",
                                 f"{accuracy[:, row, column].std(ddof=1):.3f}",
                                 f"{f1[:, row, column].mean():.4f}",
                                 f"{f1[:, row, column].std(ddof=1):.4f}"])
    print(f"Saved {output_dir / 'summary.csv'}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("datasets"))
    parser.add_argument("--out", type=Path, default=Path("data result/repeated_runs"))
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    args = parser.parse_args()
    if len(set(args.seeds)) != len(args.seeds) or len(args.seeds) < 2 or min(args.seeds) < 0:
        parser.error("Use at least two distinct nonnegative seeds.")
    repeat(args.data_dir, args.out, args.seeds)
