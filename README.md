# Clothing classification with corner markers

Junhoe Kim's PA04 experiment.

## Files

- build_datasets.py downloads Fashion-MNIST and creates the PNG folders.
- train_pa04.py trains three CNNs for 15 epochs.
- evaluate_pa04.py creates two small comparison tables.

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

Pretrained seed-42 models are included in models/seed_42/. After building the data:

    python evaluate_pa04.py

Results appear under data result/seed_42/ with the default seed.

- 01_accuracy_percent.csv gives accuracy percentages.
- 02_macro_f1.csv gives scores from 0 to 1.
- Rows are training models and columns are test conditions.

To train the three models yourself:

    python train_pa04.py

The CNN has 16- and 32-filter 3 x 3 convolution layers, max pooling, a 32-unit
dense layer and four-class softmax output. It uses Adam 0.001, batch size 64
and sparse categorical cross-entropy. Training has a fixed 15-epoch budget.
## Manually select a seed

Near the top of train_pa04.py, keep exactly one line active:

    SEED = 42
    # SEED = 43
    # SEED = 44
    # SEED = 45
    # SEED = 46

To use seed 43, comment the first line and remove # from the second line.
Save train_pa04.py before running either script. Comments next to the selection
explain the steps. All source comments are in English.

    python train_pa04.py
    python evaluate_pa04.py

Training runs the three marker conditions once for the selected seed.
Evaluation imports the selected seed from train_pa04.py automatically.
Models go to models/seed_43/ and results go to data result/seed_43/.
For another seed, change the active line and repeat these two commands.
Running the same seed again replaces that seed's model and result files.

For direct command-line selection, both scripts also support --seed:

    python train_pa04.py --seed 43
    python evaluate_pa04.py --seed 43

Use the same --seed value for both commands if overriding the commented block.
To evaluate a previously trained seed, select it and run only evaluate_pa04.py.
The dataset selection remains fixed at seed 42 in build_datasets.py.

Reference mean accuracy percentages and sample SD in percentage points:

| Training | No marker | Matching | Swapped |
| --- | ---: | ---: | ---: |
| Clean | 89.61 +/- 0.69 | 85.00 +/- 1.08 | 92.00 +/- 1.12 |
| Fixed | 84.13 +/- 2.49 | 99.98 +/- 0.00 | 24.16 +/- 13.52 |
| Random | 89.44 +/- 0.30 | 90.32 +/- 0.65 | 88.59 +/- 0.77 |

Numerical results can vary across software versions or hardware. Five training
seeds describe training variability on this dataset, not variation from new
data collection.
