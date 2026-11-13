"""Create and read the PNG datasets for the corner-marker experiment."""

import argparse
import gzip
from pathlib import Path
from urllib.request import urlopen

import numpy as np
from PIL import Image

CLASSES = ("tshirt", "shirt", "sneaker", "ankle_boot")
TRAIN = ("clean", "fixed", "random")
TEST = ("clean", "fixed", "swapped")
SOURCE_IDS = np.array([0, 6, 7, 9])
URL = "https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/"


def load_images(folder):
    files = sorted((p.name, p, label) for label, name in enumerate(CLASSES)
                   for p in (Path(folder) / name).glob("*.png"))
    if not files:
        raise ValueError(f"No PNG images in {folder}")
    images, labels = [], []
    for _, path, label in files:
        with Image.open(path) as image:
            pixels = np.array(image.convert("L"))
        if pixels.shape != (32, 32):
            raise ValueError(f"Expected a 32 x 32 image: {path}")
        images.append(pixels)
        labels.append(label)
    return np.stack(images)[..., None].astype("float32") / 255, np.array(labels)


def read_source(folder, split):
    arrays = []
    for kind, offset in (("images-idx3", 16), ("labels-idx1", 8)):
        filename = f"{split}-{kind}-ubyte.gz"
        path = folder / filename
        if not path.exists():
            with urlopen(URL + filename, timeout=60) as response:
                path.write_bytes(response.read())
        arrays.append(np.frombuffer(gzip.decompress(path.read_bytes()), "uint8", offset=offset))
    return arrays[0].reshape(-1, 28, 28), arrays[1]


def build(root):
    if (root / "train").exists() or (root / "test").exists():
        raise FileExistsError("Datasets already exist. Choose a new --data-dir to rebuild.")
    source = root / "source"
    source.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)
    corners = ((0, 0), (0, 30), (30, 0), (30, 30))
    for split, original_split in (("train", "train"), ("test", "t10k")):
        images, labels = read_source(source, original_split)
        indices = np.flatnonzero(np.isin(labels, SOURCE_IDS))
        if split == "train":
            indices = np.sort(np.concatenate([
                rng.choice(np.flatnonzero(labels == c), 600, replace=False) for c in SOURCE_IDS]))
        labels = np.searchsorted(SOURCE_IDS, labels[indices])
        markers = {"clean": np.full(len(labels), -1), "fixed": labels.copy()}
        if split == "train":
            markers["random"] = np.empty(len(labels), dtype=int)
            for label in range(4):
                markers["random"][labels == label] = rng.permutation(np.repeat(np.arange(4), 150))
        else:
            markers["swapped"] = (labels + 1) % 4
        for condition, positions in markers.items():
            for name in CLASSES:
                (root / split / condition / name).mkdir(parents=True, exist_ok=True)
            for index, label, position in zip(indices, labels, positions):
                pixels = np.pad(images[index], 2)
                if position >= 0:
                    row, column = corners[position]
                    pixels[row:row + 2, column:column + 2] = 255
                Image.fromarray(pixels).save(
                    root / split / condition / CLASSES[label] / f"image_{index:05d}.png")
            print(f"Created {split}/{condition}: {len(labels)} images", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("datasets"))
    build(parser.parse_args().data_dir)
