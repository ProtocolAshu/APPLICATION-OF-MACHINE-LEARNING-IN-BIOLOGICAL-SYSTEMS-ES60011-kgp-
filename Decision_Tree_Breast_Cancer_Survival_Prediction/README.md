# Assignment 3 — Decision Tree: Breast Cancer Survival Prediction

Predicting patient survival status from breast cancer patient records using a **Decision Tree Classifier**, and studying how tree depth affects accuracy.

## Dataset

- **File**: `Breast_Cancer.csv`
- **Records**: Breast cancer patient records with clinical and demographic attributes
- **Features**: `Age`, `Race`, `Marital Status`, `T Stage`, `N Stage`, `6th Stage`, `differentiate`, `Grade`, `A Stage`, `Tumor Size`, `Estrogen Status`, `Progesterone Status`, `Regional Node Examined`, `Reginol Node Positive`, `Survival Months`
- **Target**: `Status` (Alive = 1, Dead = 0)

## Approach

1. **Preprocessing** — map `Status` to binary; one-hot encode the 10 categorical columns.
2. **Data splitting** — 80% training / 20% testing (`random_state=42`).
3. **Depth sweep** — train Decision Trees (`criterion='gini'`) with `max_depth` from 1 to 20, treating depth as a simulated "epoch".
4. **Evaluation** — test accuracy and loss (1 − accuracy) per depth, written to `output.txt` and plotted.

## Files

| File | Description |
|------|-------------|
| `Project_3.py` | Main script — runs the depth sweep and writes results |
| `Breast_Cancer.csv` | Dataset |
| `output.txt` | Accuracy and loss per depth (epoch) table |
| `plot.png` | Accuracy and validation loss vs. tree depth |
| `Project-3_dt.docx` | Assignment report |

## How to Run

```bash
pip install pandas matplotlib scikit-learn
python Project_3.py
```

All results are written to `output.txt` and `plot.png`.

## Results

| Depth | Accuracy |
|-------|----------|
| 1–2 | 0.9118 (best) |
| 3–9 | ~0.90–0.91 |
| 10–20 | 0.86–0.89 (degrading) |

**Key insight**: accuracy peaks at shallow depths (1–2) and steadily declines beyond depth 10 — a textbook case of **overfitting**: deeper trees memorize training noise instead of generalizing. For this dataset, a shallow tree (or pruning via `ccp_alpha`) is the better choice, trading a little bias for much better generalization.
