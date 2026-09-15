# Assignment 2 — Logistic Regression: Heart Disease (Framingham) Prediction

Predicting 10-year coronary heart disease (CHD) risk from the Framingham study data using **Logistic Regression**.

## Dataset

- **File**: `framingham.csv`
- **Records**: ~4,240 patients from the Framingham Heart Study
- **Features**: 15 attributes — demographics (`male`, `age`, `education`), smoking (`currentSmoker`, `cigsPerDay`), medical history (`BPMeds`, `prevalentStroke`, `prevalentHyp`, `diabetes`), and clinical measurements (`totChol`, `sysBP`, `diaBP`, `BMI`, `heartRate`, `glucose`)
- **Target**: `TenYearCHD` (1 = developed CHD within 10 years, 0 = did not)

## Approach

1. **Missing value analysis** — several columns have gaps (notably `glucose`: 388, `education`: 105).
2. **Imputation** — median imputation via `SimpleImputer`.
3. **Feature scaling** — `StandardScaler` (mean 0, variance 1).
4. **Data splitting** — 70% training / 30% testing (`random_state=42`).
5. **Model training** — `LogisticRegression(max_iter=5000)`.
6. **Evaluation** — accuracy, confusion matrix, classification report, and coefficient analysis.

## Files

| File | Description |
|------|-------------|
| `Assignment_2.py` | Main script — runs the full pipeline |
| `framingham.csv` | Dataset |
| `output.txt` | Model evaluation results (accuracy, confusion matrix, classification report, coefficients) |
| `logistic_regression_results.csv` | Actual vs. predicted labels for the test set |
| `confusion_matrix_heatmap.png` | Confusion matrix heatmap |
| `actual_vs_predicted.png` | Actual vs. predicted CHD risk over test samples |
| `obj_data2_lr.pdf` | Assignment problem statement |

## How to Run

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
python Assignment_2.py
```

All results are written to `output.txt`.

## Results

| Metric | Value |
|--------|-------|
| Accuracy | 0.8553 |
| Confusion matrix | [[1067, 10], [174, 21]] |
| Recall (CHD class) | 0.11 |

**Key insights**:
- `age` (coef 0.576) is the strongest positive predictor, followed by `cigsPerDay` (0.316) and `sysBP` (0.274).
- The overall accuracy is high, but the model catches only ~11% of actual CHD cases — the classes are imbalanced (~15% positive), so accuracy alone is misleading. Class weighting, resampling (SMOTE), or threshold tuning would improve recall.
