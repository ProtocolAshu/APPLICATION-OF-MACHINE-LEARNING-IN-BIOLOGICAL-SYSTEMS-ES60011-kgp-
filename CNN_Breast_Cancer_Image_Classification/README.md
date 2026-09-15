# Assignment 5 — Convolutional Neural Network: Breast Cancer Image Classification

Classifying breast cancer histopathology (FNA) images as **benign** or **malignant** using a CNN built with TensorFlow / Keras.

## Dataset

- **File**: `Dataset2.zip` (extract first — creates the `Dataset2/` directory)
  - `FNA/benign/` — labeled benign images (training + validation via an 80/20 split)
  - `FNA/malignant/` — labeled malignant images
  - `test/` — unlabeled test images for prediction

## Approach

1. **Data loading** — `ImageDataGenerator` with `rescale=1./255` and a 20% validation split, fed from the directory structure.
2. **Architecture** — 3 × [Conv2D (32/64/128 filters, 3×3, ReLU) → MaxPooling (2×2)] → Flatten → Dense(128, ReLU) → Dropout(0.5) → Dense(1, sigmoid).
3. **Training** — Adam optimizer (lr = 0.001), binary cross-entropy loss, 10 epochs, batch size 32.
4. **Evaluation** — validation accuracy/loss per epoch, confusion matrix, and classification report on the validation set; predictions on the unlabeled test images.

## Files

| File | Description |
|------|-------------|
| `main.py` | Main script — training, evaluation, prediction |
| `Dataset2.zip` | Dataset archive |
| `accuracy_plot.png` | Training vs. validation accuracy per epoch |
| `loss_plot.png` | Training vs. validation loss per epoch |
| `OBJ_DATA2.pdf` | Assignment problem statement |

## How to Run

```bash
pip install numpy pandas matplotlib tensorflow scikit-learn

# Extract the dataset first
unzip Dataset2.zip

python main.py
```

Running the script also generates `output.txt` (model summary, per-epoch metrics, validation evaluation, confusion matrix, classification report, test-image predictions) and the trained model `model_cnn.h5` — neither is committed to the repo.

## Model Architecture

```
Conv2D(32, 3×3, relu) → MaxPooling(2×2)
Conv2D(64, 3×3, relu) → MaxPooling(2×2)
Conv2D(128, 3×3, relu) → MaxPooling(2×2)
Flatten → Dense(128, relu) → Dropout(0.5) → Dense(1, sigmoid)
```

Images are resized to 150×150×3. The increasing filter counts let the network learn progressively more abstract features, and dropout combats overfitting on the modest dataset size.
