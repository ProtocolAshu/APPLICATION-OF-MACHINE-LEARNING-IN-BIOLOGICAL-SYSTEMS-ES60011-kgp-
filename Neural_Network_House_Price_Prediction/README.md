# Assignment 4 — Neural Network (from scratch): House Price Prediction

Implementing a feedforward neural network **from scratch in NumPy** (no deep-learning framework) for regression on the Boston housing dataset, with k-fold cross-validation over hyperparameters.

## Dataset

- **File**: `housing.csv` (Boston housing derived — `MEDV` in dollars)
- **Records**: 489 rows, 4 columns
- **Features**: `RM` (rooms per dwelling), `LSTAT` (lower-status population %), `PTRATIO` (pupil-teacher ratio)
- **Target**: `MEDV` (median house price)

## Approach

1. **Preprocessing** — select 3 features and the target; standardize both X and y (zero mean, unit variance).
2. **Network architecture** — input layer (3) → one hidden layer (sigmoid activation) → linear output layer (1), since this is regression.
3. **Backpropagation** — hand-derived forward/backward passes with gradient-descent weight updates, implemented from scratch.
4. **Training configurations** — as required by the assignment:
   | Case | Hidden neurons | Learning rate |
   |------|----------------|---------------|
   | (a) | 3 | 0.01 |
   | (b) | 4 | 0.001 |
   | (c) | 5 | 0.0001 |
5. **Cross-validation** — k-fold CV (k = 3, 5, 10 respectively) reporting the average test loss per configuration.

## Files

| File | Description |
|------|-------------|
| `P4_code.py` | Main script — full implementation; writes all output to `output.txt` |
| `NN.ipynb` | Tutorial notebook on the backpropagation worked example (by the course TA) |
| `housing.csv` | Dataset |
| `output.txt` | Code listing + execution results (data inspection, training losses, CV results) |
| `P4_NN.docx` | Assignment report |

## How to Run

```bash
pip install pandas numpy scikit-learn
python P4_code.py
```

All results are written to `output.txt`.

## Results

The output file contains, for each configuration: the loss every 100 epochs during training, plus the average k-fold CV test loss. The lowest learning rate (case c, lr = 0.0001) converges slowly — after 1,000 epochs the network is still far from its optimum, visibly in the loss curve — while lr = 0.01 (case a) makes the fastest progress, illustrating the learning-rate trade-off between convergence speed and stability.
