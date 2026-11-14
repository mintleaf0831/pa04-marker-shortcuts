"""Create and read the PNG datasets for the corner-marker experiment."""

import argparse
import gzip
from pathlib import Path
from urllib.request import urlopen

import numpy as np
from PIL import Image

# Use the same class order when making data, training, and testing.
CLASSES = ("tshirt", "shirt", "sneaker", "ankle_boot")
TRAIN_CONDITIONS = ("clean", "fixed", "random")
TEST_CONDITIONS = ("clean", "fixed", "swapped")
CLASS_IDS = np.array([0, 6, 7, 9])
URL = "https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/"


def load_data(folder):
    # Read each class folder and keep the images in a fixed order.
    files = sorted((p.name, p, label) for label, name in enumerate(CLASSES) for p in (Path(folder) / name).glob("*.png"))
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

    # Scale pixel values from 0-255 to 0-1. Add one channel for gray images.
    return np.stack(images)[..., None].astype("float32") / 255, np.array(labels)


def read_source(folder, split):
    arrays = []

    # Download missing source files. Skip the 16-byte image header or 8-byte label header.
    for kind, offset in (("images-idx3", 16), ("labels-idx1", 8)):
        filename = f"{split}-{kind}-ubyte.gz"
        path = folder / filename
        if not path.exists():
            with urlopen(URL + filename, timeout=60) as response:
                path.write_bytes(response.read())
        arrays.append(np.frombuffer(gzip.decompress(path.read_bytes()), "uint8", offset=offset))

    return arrays[0].reshape(-1, 28, 28), arrays[1]


def build_data(root):
    # Stop if data already exists, so we do not replace it by mistake.
    if (root / "train").exists() or (root / "test").exists():
        raise FileExistsError("Datasets already exist. Choose a new --data-dir to rebuild.")
    source = root / "source"
    source.mkdir(parents=True, exist_ok=True)

    # Keep the chosen images and marker positions the same each time we build the data.
    rng = np.random.default_rng(42)
    corners = ((0, 0), (0, 30), (30, 0), (30, 30))

    # Use the official training and test sets separately.
    for split, original_split in (("train", "train"), ("test", "t10k")):
        images, labels = read_source(source, original_split)
        indices = np.flatnonzero(np.isin(labels, CLASS_IDS))

        # Choose 600 images per class for training. Keep all test images from these classes.
        if split == "train":
            indices = np.sort(np.concatenate([ rng.choice(np.flatnonzero(labels == c), 600, replace=False) for c in CLASS_IDS]))

        # Change the original class numbers (0, 6, 7, 9) to 0, 1, 2, 3.
        labels = np.searchsorted(CLASS_IDS, labels[indices])

        # -1 means no marker. In the fixed condition, the class sets the corner.
        markers = {"clean": np.full(len(labels), -1), "fixed": labels.copy()}
        if split == "train":
            # Within each class, use each corner 150 times and mix their order.
            markers["random"] = np.empty(len(labels), dtype=int)
            for label in range(4):
                markers["random"][labels == label] = rng.permutation(np.repeat(np.arange(4), 150))
        else:
            # Move each test marker to the next class corner. Keep the true label unchanged.
            markers["swapped"] = (labels + 1) % 4
        for condition, positions in markers.items():
            for name in CLASSES:
                (root / split / condition / name).mkdir(parents=True, exist_ok=True)
            for index, label, position in zip(indices, labels, positions):
                # Add a black border so the white marker does not cover the clothing.
                pixels = np.pad(images[index], 2)
                if position >= 0:
                    row, column = corners[position]
                    pixels[row:row + 2, column:column + 2] = 255

                # Save each image in its true class folder, using its original image number.
                Image.fromarray(pixels).save( root / split / condition / CLASSES[label] / f"image_{index:05d}.png")
            print(f"Created {split}/{condition}: {len(labels)} images", flush=True)


if __name__ == "__main__":
    # Allow a different data folder to be chosen from the command line.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("datasets"))

    build_data(parser.parse_args().data_dir)
