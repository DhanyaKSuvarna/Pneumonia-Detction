# Pneumonia Detection from Chest X-Rays — CNN vs. Transfer Learning

A course project for **Neural Networks and Deep Learning**: binary classification of chest
X-rays into `NORMAL` / `PNEUMONIA`, comparing a convolutional neural network trained from
scratch against a transfer-learning model built on a pretrained ResNet18 backbone.

## Contents

| File | Description |
|---|---|
| `pneumonia_detection_cnn_vs_transfer_learning.ipynb` | Main notebook: data loading, both models, training, evaluation, and comparison |
| `README.md` | This file |

## Dataset

**Chest X-Ray Images (Pneumonia)** — Kermany et al., hosted on Kaggle:
https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia

(Mirror: https://data.mendeley.com/datasets/rscbjbr9sj/3)

No API key is required — the dataset is downloaded manually:

1. Open the Kaggle link above and sign in (free account).
2. Click **Download** (top-right) to get `archive.zip` (~1.2 GB).
3. Place the zip file in the same folder as the notebook.
4. Run the first few cells of the notebook — they extract it automatically into
   `data/chest_xray/{train,test,val}/{NORMAL,PNEUMONIA}/`.

The dataset is **not** committed to this repository (too large for Git); each user downloads
their own local copy as above.

## Approach

1. **Preprocessing** — images resized to 224x224, converted to 3-channel tensors, normalized
   with ImageNet statistics; light augmentation (flip, small rotation) on the training split.
   A stratified 85/15 split is carved out of the training folder for validation (the dataset's
   own `val/` folder only has 16 images), and a weighted sampler corrects for class imbalance.
2. **Model 1 — `SimpleCNN`**: a 4-block `Conv → BatchNorm → ReLU → MaxPool` network trained
   entirely from scratch, with a small dropout-regularized classifier head.
3. **Model 2 — Transfer learning**: ResNet18 pretrained on ImageNet, convolutional backbone
   frozen, with a new fully-connected head trained on top.
4. **Training** — both models use the same loss (cross-entropy), optimizer (Adam), learning
   rate, and number of epochs, so any difference in results reflects the modeling approach
   rather than the training setup.
5. **Evaluation & comparison** — accuracy, precision, recall, F1, ROC-AUC, confusion matrices,
   ROC curves, and training curves are compared side by side, followed by a written discussion
   of why the two approaches diverge (convergence speed, overfitting, parameter efficiency,
   recall on the clinically important PNEUMONIA class).

## Requirements

```
torch
torchvision
numpy
pandas
matplotlib
scikit-learn
```

Install with:

```bash
pip install torch torchvision numpy pandas matplotlib scikit-learn
```

A GPU (e.g. via Google Colab or Kaggle Notebooks) is recommended for reasonable training time.

## How to run

1. Clone this repository.
2. Download the dataset as described above and place the zip next to the notebook.
3. Open `pneumonia_detection_cnn_vs_transfer_learning.ipynb` in Jupyter / Colab / Kaggle.
4. Run all cells top to bottom.

## Results

Test-set performance (1,248 images: 468 NORMAL, 780 PNEUMONIA), 10 epochs, both models trained
under identical conditions:

| Metric | SimpleCNN (scratch) | ResNet18 (transfer) |
|---|---|---|
| Accuracy | 83.81% | 90.22% |
| Precision | 80.17% | 89.45% |
| Recall | 98.46% | 95.64% |
| F1 | 88.38% | 92.44% |
| ROC-AUC | 92.85% | 95.34% |
| Trainable params | 405,954 | 65,922 |

**Summary:** the ResNet18 transfer-learning model outperformed the from-scratch CNN on every
metric except recall, while training ~6x fewer parameters. Both models achieve very high
PNEUMONIA recall (98.46% and 95.64% respectively) — clinically the more important number, since
a missed pneumonia case is costlier than a false alarm — but SimpleCNN's higher recall comes at
the cost of much lower NORMAL-class precision (it over-predicts PNEUMONIA more often, per its
lower overall accuracy and precision). ResNet18 gives a more balanced trade-off between the two
classes, consistent with it benefiting from ImageNet-pretrained features that a small
from-scratch CNN cannot learn from ~8,800 training images alone.

## License

Course project — for academic use.
