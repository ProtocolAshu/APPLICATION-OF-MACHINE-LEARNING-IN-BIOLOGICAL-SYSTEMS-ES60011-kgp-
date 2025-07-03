import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
import sys

# Informing user that output will be stored in output.txt
print("ALL OUTPUT ARE IN output.txt FILE \n")

# Redirecting output to output.txt
output_file = open('output.txt', 'w')
sys.stdout = output_file

def print_code_block(code_str):
    """
    This function prints the code inside a rectangle box format to make it more readable in the output.txt file.
    """
    border = "+" + "-" * (len(max(code_str.split('\n'), key=len)) + 2) + "+"
    output_file.write(border + "\n")
    for line in code_str.split('\n'):
        output_file.write("| " + line + " " * (len(border) - len(line) - 3) + "|\n")
    output_file.write(border + "\n\n")


# SECTION 1: Load and Inspect the Dataset
output_file.write("# SECTION 1: Loading the Boston Housing Dataset\n\n")
code_section_1 = """
file_path = 'housing.csv'
try:
    housing_data = pd.read_csv(file_path)
    print("File loaded successfully!\\n")
except FileNotFoundError:
    print(f"Error: The file '{file_path}' was not found.\\n")

# Displaying the first few rows and info
print("First few rows of the dataset:\\n")
print(housing_data.head(), "\\n")

print("Dataset Information:\\n")
housing_data.info()
print("\\n")
"""
print_code_block(code_section_1)

# Code Execution for Section 1
file_path = 'housing.csv'
try:
    housing_data = pd.read_csv(file_path)
    print("File loaded successfully!\n")
except FileNotFoundError:
    print(f"Error: The file '{file_path}' was not found.\n")

# Displaying the first few rows and info
print("First few rows of the dataset:\n")
print(housing_data.head(), "\n")

print("Dataset Information:\n")
housing_data.info()
print("\n")

# SECTION 2: Preprocessing the Data
output_file.write("# SECTION 2: Preprocessing the Data\n\n")
code_section_2 = """
# Splitting dataset into features and target
X = housing_data[['RM', 'LSTAT', 'PTRATIO']].values  # Input features
y = housing_data['MEDV'].values.reshape(-1, 1)  # Target values

# Standardize the input data
X_mean = np.mean(X, axis=0)
X_std = np.std(X, axis=0)
X = (X - X_mean) / X_std

# Standardize the target values
y_mean = np.mean(y, axis=0)
y_std = np.std(y, axis=0)
y = (y - y_mean) / y_std

print("Input features (first 5 rows):\\n", X[:5], "\\n")
print("Target values (first 5 rows):\\n", y[:5], "\\n")
"""
print_code_block(code_section_2)

# Code Execution for Section 2
# Splitting dataset into features and target
X = housing_data[['RM', 'LSTAT', 'PTRATIO']].values  # Input features
y = housing_data['MEDV'].values.reshape(-1, 1)  # Target values

# Standardize the input data
X_mean = np.mean(X, axis=0)
X_std = np.std(X, axis=0)
X = (X - X_mean) / X_std

# Standardize the target values
y_mean = np.mean(y, axis=0)
y_std = np.std(y, axis=0)
y = (y - y_mean) / y_std

print("Input features (first 5 rows):\n", X[:5], "\n")
print("Target values (first 5 rows):\n", y[:5], "\n")

# SECTION 3: Neural Network Class Definition
output_file.write("# SECTION 3: Defining the Neural Network\n\n")
code_section_3 = """
class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, learning_rate):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate

        # Initialize weights with random values
        self.W1 = np.random.randn(self.input_size, self.hidden_size) * 0.01
        self.b1 = np.zeros((1, self.hidden_size))
        self.W2 = np.random.randn(self.hidden_size, self.output_size) * 0.01
        self.b2 = np.zeros((1, self.output_size))

    # Activation function: Sigmoid
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    # Derivative of Sigmoid
    def sigmoid_derivative(self, z):
        return z * (1 - z)

    # Forward propagation
    def forward(self, X):
        self.Z1 = np.dot(X, self.W1) + self.b1
        self.A1 = self.sigmoid(self.Z1)
        self.Z2 = np.dot(self.A1, self.W2) + self.b2
        self.A2 = self.Z2  # Linear activation for output layer (since this is regression)
        return self.A2

    # Backward propagation and weight updates
    def backward(self, X, y, output):
        m = X.shape[0]  # Number of samples

        # Calculate error (mean squared error derivative)
        dZ2 = output - y
        dW2 = np.dot(self.A1.T, dZ2) / m
        db2 = np.sum(dZ2, axis=0, keepdims=True) / m

        # Backpropagate through hidden layer
        dA1 = np.dot(dZ2, self.W2.T)
        dZ1 = dA1 * self.sigmoid_derivative(self.A1)
        dW1 = np.dot(X.T, dZ1) / m
        db1 = np.sum(dZ1, axis=0, keepdims=True) / m

        # Update weights and biases
        self.W1 -= self.learning_rate * dW1
        self.b1 -= self.learning_rate * db1
        self.W2 -= self.learning_rate * dW2
        self.b2 -= self.learning_rate * db2

    # Training the network
    def train(self, X, y, epochs):
        for epoch in range(epochs):
            output = self.forward(X)
            self.backward(X, y, output)
            if epoch % 100 == 0:
                loss = np.mean(np.square(output - y))  # Mean squared error
                print(f'Epoch {epoch}, Loss: {loss}')
        print("\\n")
"""
print_code_block(code_section_3)

# Neural Network Class Definition (actual code execution)
class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, learning_rate):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        
        # Initialize weights with random values
        self.W1 = np.random.randn(self.input_size, self.hidden_size) * 0.01
        self.b1 = np.zeros((1, self.hidden_size))
        self.W2 = np.random.randn(self.hidden_size, self.output_size) * 0.01
        self.b2 = np.zeros((1, self.output_size))
    
    # Activation function: Sigmoid
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    # Derivative of Sigmoid
    def sigmoid_derivative(self, z):
        return z * (1 - z)
    
    # Forward propagation
    def forward(self, X):
        self.Z1 = np.dot(X, self.W1) + self.b1
        self.A1 = self.sigmoid(self.Z1)
        self.Z2 = np.dot(self.A1, self.W2) + self.b2
        self.A2 = self.Z2  # Linear activation for output layer (since this is regression)
        return self.A2
    
    # Backward propagation and weight updates
    def backward(self, X, y, output):
        m = X.shape[0]  # Number of samples
        
        # Calculate error (mean squared error derivative)
        dZ2 = output - y
        dW2 = np.dot(self.A1.T, dZ2) / m
        db2 = np.sum(dZ2, axis=0, keepdims=True) / m
        
        dA1 = np.dot(dZ2, self.W2.T)
        dZ1 = dA1 * self.sigmoid_derivative(self.A1)
        dW1 = np.dot(X.T, dZ1) / m
        db1 = np.sum(dZ1, axis=0, keepdims=True) / m
        
        self.W1 -= self.learning_rate * dW1
        self.b1 -= self.learning_rate * db1
        self.W2 -= self.learning_rate * dW2
        self.b2 -= self.learning_rate * db2
    
    # Training the network
    def train(self, X, y, epochs):
        for epoch in range(epochs):
            output = self.forward(X)  # Forward pass
            self.backward(X, y, output)  # Backward pass (update weights)
            
            # Print loss every 100 epochs
            if epoch % 100 == 0:
                loss = np.mean(np.square(output - y))  # Mean squared error
                print(f'Epoch {epoch}, Loss: {loss}')
        print("\n")

# SECTION 4: Training the Neural Network
output_file.write("# SECTION 4: Training the Neural Network\n")
code_section_4 = """
# Code:
# Parameters
input_size = X.shape[1]  # 3 features
output_size = 1  # Regression output (house price)

# Train the model with case (a): Hidden neurons = 3, learning rate = 0.01
print("Training with 3 neurons in the hidden layer and learning rate 0.01:\\n")
nn = NeuralNetwork(input_size, hidden_size=3, output_size=output_size, learning_rate=0.01)
nn.train(X, y, epochs=1000)
"""
print_code_block(code_section_4)

# Parameters
input_size = X.shape[1]  # 3 features
output_size = 1  # Regression output (house price)

# Train the model with case (a): Hidden neurons = 3, learning rate = 0.01
print("Training with 3 neurons in the hidden layer and learning rate 0.01:\n")
nn = NeuralNetwork(input_size, hidden_size=3, output_size=output_size, learning_rate=0.01)
nn.train(X, y, epochs=1000)

# SECTION 5: Cross-Validation
output_file.write("# SECTION 5: Cross-Validation\n")
code_section_5 = """
# Code:
def cross_validation(X, y, hidden_size, learning_rate, epochs, k=5):
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    fold_losses = []

    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # Initialize a new network for each fold
        nn = NeuralNetwork(input_size=X.shape[1], hidden_size=hidden_size, output_size=1, learning_rate=learning_rate)

        # Train the model
        nn.train(X_train, y_train, epochs=epochs)

        # Evaluate the loss on the test set
        test_output = nn.forward(X_test)
        test_loss = np.mean(np.square(test_output - y_test))
        fold_losses.append(test_loss)

    average_loss = np.mean(fold_losses)
    print(f"Average loss for {k}-fold CV (hidden neurons = {hidden_size}, lr = {learning_rate}): {average_loss}\\n")
    """
print_code_block(code_section_5)

def cross_validation(X, y, hidden_size, learning_rate, epochs, k=5):
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    fold_losses = []
    
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        # Initialize a new network for each fold
        nn = NeuralNetwork(input_size=X.shape[1], hidden_size=hidden_size, output_size=1, learning_rate=learning_rate)
        
        # Train the model
        nn.train(X_train, y_train, epochs=epochs)
        
        # Evaluate the loss on the test set
        test_output = nn.forward(X_test)
        test_loss = np.mean(np.square(test_output - y_test))
        fold_losses.append(test_loss)
    
    average_loss = np.mean(fold_losses)
    print(f"Average loss for {k}-fold CV (hidden neurons = {hidden_size}, lr = {learning_rate}): {average_loss}\n")

# Cross-validation for case (a): hidden neurons = 3, learning rate = 0.01
print("Cross-validation with 3 neurons in the hidden layer and learning rate 0.01:\n")
cross_validation(X, y, hidden_size=3, learning_rate=0.01, epochs=1000, k=3)

# Cross-validation for case (b): hidden neurons = 4, learning rate = 0.001
print("Cross-validation with 4 neurons in the hidden layer and learning rate 0.001:\n")
cross_validation(X, y, hidden_size=4, learning_rate=0.001, epochs=1000, k=5)

# Cross-validation for case (c): hidden neurons = 5, learning rate = 0.0001
print("Cross-validation with 5 neurons in the hidden layer and learning rate 0.0001:\n")
cross_validation(X, y, hidden_size=5, learning_rate=0.0001, epochs=1000, k=10)

# Close the output file after writing all outputs
output_file.close()
