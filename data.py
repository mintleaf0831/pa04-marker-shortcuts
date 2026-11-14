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
    files = []
    for label, name in enumerate(CLASSES):
        class_folder = Path(folder) / name
        for path in class_folder.glob("*.png"):
            files.append((path.name, path, label))
    files.sort()
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
    images = np.stack(images)
    images = images.reshape(len(images), 32, 32, 1)
    images = images.astype("float32") / 255
    labels = np.array(labels)
    return images, labels


def read_source(folder, split):
    arrays = []

    # Download missing source files. Skip the 16-byte image header or 8-byte label header.
    for kind, offset in (("images-idx3", 16), ("labels-idx1", 8)):
        filename = f"{split}-{kind}-ubyte.gz"
        path = folder / filename
        if not path.exists():
            with urlopen(URL + filename, timeout=60) as response:
                path.write_bytes(response.read())

        compressed_data = path.read_bytes()
        raw_data = gzip.decompress(compressed_data)
        values = np.frombuffer(raw_data, dtype="uint8", offset=offset)
        arrays.append(values)

    images = arrays[0].reshape(-1, 28, 28)
    labels = arrays[1]
    return images, labels


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
        # Choose 600 images per class for training. Use all selected-class test images.
        selected = []
        for class_id in CLASS_IDS:
            class_indices = np.where(labels == class_id)[0]
            if split == "train":
                class_indices = rng.choice(class_indices, 600, replace=False)
            selected.extend(class_indices)
        indices = np.array(sorted(selected))

        # Change the original class numbers (0, 6, 7, 9) to 0, 1, 2, 3.
        selected_labels = labels[indices]
        labels = np.zeros(len(indices), dtype=int)
        for new_label, class_id in enumerate(CLASS_IDS):
            labels[selected_labels == class_id] = new_label

        # -1 means no marker. In the fixed condition, the class sets the corner.
        markers = {"clean": np.full(len(labels), -1), "fixed": labels.copy()}
        if split == "train":
            # Within each class, use each corner 150 times and mix their order.
            markers["random"] = np.empty(len(labels), dtype=int)
            for label in range(4):
                positions = np.repeat([0, 1, 2, 3], 150)
                positions = rng.permutation(positions)
                markers["random"][labels == label] = positions
        else:
            # Move each test marker to the next class corner. Keep the true label unchanged.
            markers["swapped"] = (labels + 1) % 4
        for condition, positions in markers.items():
            for name in CLASSES:
                class_folder = root / split / condition / name
                class_folder.mkdir(parents=True, exist_ok=True)
            for index, label, position in zip(indices, labels, positions):
                # Add a black border so the white marker does not cover the clothing.
                pixels = np.pad(images[index], 2)
                if position >= 0:
                    row, column = corners[position]
                    pixels[row:row + 2, column:column + 2] = 255

                # Save each image in its true class folder, using its original image number.
                filename = f"image_{index:05d}.png"
                image_path = root / split / condition / CLASSES[label] / filename
                image = Image.fromarray(pixels)
                image.save(image_path)
            print(f"Created {split}/{condition}: {len(labels)} images", flush=True)


if __name__ == "__main__":
    # By default, use the datasets folder next to this file.
    # A custom --data-dir path is relative to the terminal folder unless it is absolute.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parent / "datasets")

    args = parser.parse_args()
    build_data(args.data_dir)
