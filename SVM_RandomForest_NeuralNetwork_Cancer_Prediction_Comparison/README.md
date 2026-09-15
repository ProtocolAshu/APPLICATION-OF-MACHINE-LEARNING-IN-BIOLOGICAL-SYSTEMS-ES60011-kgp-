# Assignment 6 — Model Comparison: SVM vs. Random Forest vs. Neural Network on Cancer Prediction

Training and comparing three classic ML models — **SVM**, **Random Forest**, and a **Neural Network** — for classifying breast tumors as benign (B) or malignant (M).

## Dataset

- **File**: `breast cancer.csv` (Wisconsin Diagnostic Breast Cancer)
- **Records**: 569 samples, 30 numeric features computed from digitized fine-needle-aspirate images (radius, texture, perimeter, area, smoothness, … each with mean/se/worst variants)
- **Target**: `diagnosis` — B (benign) or M (malignant)

## Approach

1. **Preprocessing** — mean-impute missing values; 80/20 train/test split (`random_state=42`).
2. **SVM** — RBF kernel inside a `Pipeline` with `StandardScaler` (scaling is essential for SVMs).
3. **Random Forest** — 100 trees.
4. **Neural Network** — `MLPClassifier` with grid search over `hidden_layer_sizes` [(10,), (50,), (100,)] × `activation` [tanh, relu], 5-fold CV.
5. **Comparison** — accuracy + classification report for all three models; results stored as CSV and bar chart.

## Files

| File | Description |
|------|-------------|
| `22CS30009_P6.py` | Main script — trains all three models and writes the comparison |
| `breast cancer.csv` | Dataset |
| `output.txt` | Full output: dataset description, per-model accuracy and classification reports, discussion, conclusion |
| `model_performance.csv` | Accuracy per model |
| `model_performance.png` | Bar chart comparing model accuracies |
| `p-6.docx` | Assignment report |

## How to Run

```bash
pip install pandas numpy matplotlib scikit-learn
python 22CS30009_P6.py
```

All results are written to `output.txt`, `model_performance.csv`, and `model_performance.png`.

## Results

| Model | Accuracy |
|-------|----------|
| **SVM (RBF kernel)** | **0.982** |
| Random Forest | 0.965 |
| Neural Network (MLP) | 0.623 |

**Conclusion**: SVM with the RBF kernel is the clear winner here — tumor morphology features are highly separable in a scaled kernel space. The MLP underperforms badly (62%) even after grid-search tuning (best: 10 hidden neurons, tanh), largely because the raw features span very different scales and no scaling was applied for the network path — a useful reminder that preprocessing choices (like feature scaling) can matter as much as model choice.
