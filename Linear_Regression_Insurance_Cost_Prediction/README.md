# Assignment 1 — Linear Regression: Insurance Cost Prediction

Predicting medical insurance charges from patient attributes using **Linear Regression** (scikit-learn).

## Dataset

- **File**: `data_insurance.csv`
- **Records**: 1,338 patients with 7 attributes
- **Features**: `age`, `sex`, `bmi`, `children`, `smoker`, `region`
- **Target**: `charges` (annual insurance cost in USD)

## Approach

1. **Data exploration** — inspect structure, summary statistics, and missing values (none found).
2. **Feature engineering** — label-encode the categorical columns (`sex`, `smoker`, `region`).
3. **Data splitting** — 80% training / 20% testing (`random_state=42`).
4. **Model training** — `sklearn.linear_model.LinearRegression`.
5. **Evaluation** — MSE, RMSE, and R² on the test set; coefficient analysis for feature importance.

## Files

| File | Description |
|------|-------------|
| `Project_1.py` | Main script — runs the full analysis and writes the report |
| `data_insurance.csv` | Dataset |
| `insurance_analysis_output.txt` | Full formatted analysis report |
| `actual_vs_predicted.png` | Scatter plot of actual vs. predicted charges with trend line |
| `project-1_linearregression.pdf` | Assignment problem statement |

## How to Run

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
python Project_1.py
```

All results are written to `insurance_analysis_output.txt`.

## Results

| Metric | Value |
|--------|-------|
| MSE | 33,635,210.43 |
| RMSE | 5,799.59 |
| R² | 0.7833 |

**Key insight**: `smoker` is by far the strongest predictor of insurance charges — smokers are charged substantially more. The model explains roughly 78% of the variance in charges; the deviations visible in `actual_vs_predicted.png` suggest non-linear terms (e.g., BMI × smoker) would improve fit.

## Possible Refinements

- Polynomial / interaction features
- Regularized regression (Lasso / Ridge)
- Non-linear models (Random Forest, Gradient Boosting)
- K-fold cross-validation
