# Neural Network for Boston Housing Dataset

This project implements a neural network for regression to predict house prices using the Boston Housing Dataset. The code handles data preprocessing, training of a neural network, and cross-validation. All outputs are saved to an `output.txt` file for easy reference.

## Prerequisites

Ensure the following libraries are installed:

- **Python 3.x**
- **Pandas**
- **Numpy**
- **Scikit-learn**

You can install the required libraries using:
    pip install pandas numpy scikit-learn


*Files*
1. P4_code.py: This is the main code file that runs the neural network.

2. housing.csv: The dataset used for training the model (ensure this file is in the same directory).

3. output.txt: The output file where all code output and results will be saved.

*How to Execute*
1. Download the Dataset: Place the housing.csv file in the same directory as P4_code.py.

2. Run the Code:
      Execute the Python script by running the following command in your terminal or command prompt:
                # python P4_code.py #

3. View Results:

After the execution is complete, open the output.txt file to view all the results, including:

    a. Data loading and initial inspection
    b. Preprocessing details (standardized features and targets)
    c. Neural network training results with loss values during each epoch
    d. Cross-validation results with the average loss for each set of hyperparameters.


*Code Structure*
1. Data Loading:
   Loads the housing.csv file using Pandas and inspects the dataset.

2. Preprocessing:
   Selects three features (RM, LSTAT, PTRATIO) and the target (MEDV).
   
   Standardizes both input features and target values.

3. Neural Network:
   A simple feedforward neural network with one hidden layer and sigmoid activation.

   Custom implementation of forward propagation, backward propagation, and weight updates.

4. Training:
   The model is trained using different configurations of hidden neurons and learning rates, printing the loss every 100 epochs.

5. Cross-Validation:
   Implements k-fold cross-validation to evaluate the model performance for different configurations of hidden neurons and learning rates.

*Example Outputs*
All outputs, including code execution results, will be saved in the output.txt file. The file will contain:

1. Data inspection (head and info)
2. Preprocessing results (standardized features)
3. Training results (loss per epoch)
4. Cross-validation performance (average loss)