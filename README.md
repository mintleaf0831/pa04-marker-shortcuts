# Clothing classification with corner markers

Junhoe Kim's PA04 experiment.

## Files

- build_datasets.py downloads Fashion-MNIST and creates the PNG folders.
- train_pa04.py trains three CNNs for 15 epochs.
- evaluate_pa04.py creates two small comparison tables.
- run_repeats.py repeats training with five seeds and reports mean and sample SD.

## Setup

    git clone https://github.com/mintleaf0831/pa04-marker-shortcuts.git
    cd pa04-marker-shortcuts
    conda env create -f environment.yml
    conda activate pa04-markers

The recorded runs used Python 3.12.14, TensorFlow 2.20.0, NumPy 2.2.6 and
Pillow 13.0.0, with CPU execution on Windows 11. A CPU is sufficient.

## Data and experiment

    python build_datasets.py

This downloads the original data from the official Fashion-MNIST repository
and recreates the PNG datasets. Existing image folders are not overwritten.
To rebuild separately, use --data-dir datasets_new.

The four classes are T-shirt/top, shirt, sneaker and ankle boot. Training uses
the same 2,400 images in all conditions, with 600 per class. Testing uses all
4,000 images of these classes from the separate official test split.
Data selection uses seed 42.

Each 28 x 28 image is centered in a 32 x 32 black canvas. White 2 x 2 markers
lie entirely in the border.

- clean training uses no marker.
- fixed training maps classes to top left, top right, bottom left, bottom right.
- random training uses each corner exactly 150 times within each true class.
- Test conditions remove the marker, match the fixed rule, or use the next
  class's corner in the above order, cycling the last class back to the first.

All labels are correct. Data and marker assignments stay fixed across runs.
The data source, paper and third-party license are available here:

https://github.com/zalandoresearch/fashion-mnist
https://arxiv.org/abs/1708.07747

## Evaluate or train

Pretrained seed-42 models are included in models/. After building the data:

    python evaluate_pa04.py

Results appear under data result/evaluation/.

- 01_accuracy_percent.csv gives accuracy percentages.
- 02_macro_f1.csv gives scores from 0 to 1.
- Rows are training models and columns are test conditions.

To train the three models yourself:

    python train_pa04.py

The CNN has 16- and 32-filter 3 x 3 convolution layers, max pooling, a 32-unit
dense layer and four-class softmax output. It uses Adam 0.001, batch size 64
and sparse categorical cross-entropy. Training has a fixed 15-epoch budget.
Training replaces model files in its output folder. For a separate run:

    python train_pa04.py --seed 43 --out models/seed_43

## Repeat five seeds

    python run_repeats.py

This retrains all models with seeds 42-46 and evaluates them. Results are
written to data result/repeated_runs/summary.csv. Training and test data stay
fixed. Only initialization and shuffling vary. Sample SD uses n-1.

Reference mean accuracy percentages and sample SD in percentage points:

| Training | No marker | Matching | Swapped |
| --- | ---: | ---: | ---: |
| Clean | 89.61 +/- 0.69 | 85.00 +/- 1.08 | 92.00 +/- 1.12 |
| Fixed | 84.13 +/- 2.49 | 99.98 +/- 0.00 | 24.16 +/- 13.52 |
| Random | 89.44 +/- 0.30 | 90.32 +/- 0.65 | 88.59 +/- 0.77 |

Numerical results can vary across software versions or hardware. Five training
seeds describe training variability on this dataset, not variation from new
data collection.
