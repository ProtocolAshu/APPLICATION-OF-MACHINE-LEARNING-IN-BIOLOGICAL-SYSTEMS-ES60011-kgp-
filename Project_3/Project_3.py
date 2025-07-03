import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Load the dataset (replace 'Breast_Cancer.csv' with your actual dataset file path)
data = pd.read_csv('Breast_Cancer.csv')

# Convert 'Status' column to binary (Dead -> 0, Alive -> 1)
data['Status'] = data['Status'].map({'Dead': 0, 'Alive': 1})

# List of categorical columns to convert using one-hot encoding
categorical_columns = ['Race', 'Marital Status', 'T Stage ', 'N Stage', '6th Stage', 
                       'differentiate', 'Grade', 'A Stage', 'Estrogen Status', 'Progesterone Status']

# One-hot encoding for categorical columns
data = pd.get_dummies(data, columns=categorical_columns)

# Separate the features (X) and target (y)
X = data.drop(columns=['Status'])
y = data['Status']

# Split the dataset into training (80%) and testing (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Lists to store accuracy and loss over different depths (simulated epochs)
accuracies = []
losses = []

# Train the model with different depths (1 to 20 to simulate epochs)
for epoch in range(1, 21):
    # Create a Decision Tree model with max_depth as epoch (to simulate iterations)
    clf = DecisionTreeClassifier(criterion='gini', max_depth=epoch, random_state=42)
    
    # Fit the model on training data
    clf.fit(X_train, y_train)
    
    # Predict on test data
    y_pred = clf.predict(X_test)
    
    # Calculate accuracy
    acc = accuracy_score(y_test, y_pred)
    accuracies.append(acc)
    
    # Simulate loss as (1 - accuracy)
    loss = 1 - acc
    losses.append(loss)

# Write the accuracy and loss results to output.txt in a well-formatted way
with open('output.txt', 'w') as f:
    f.write("="*40 + "\n")
    f.write("Performance Metrics Over Epochs (Tree Depth)\n")
    f.write("="*40 + "\n")
    f.write(f"{'Epoch':^10} | {'Accuracy':^15} | {'Loss':^10}\n")
    f.write("-"*40 + "\n")
    for epoch in range(1, 21):
        f.write(f"{epoch:^10} | {accuracies[epoch-1]:^15.4f} | {losses[epoch-1]:^10.4f}\n")
    f.write("-"*40 + "\n")
    
    # Write final model accuracy at the highest depth (epoch 20)
    f.write(f'\nFinal Accuracy at depth 20: {accuracies[-1]:.4f}\n')
    f.write("="*40 + "\n")

# Plotting accuracy and validation loss over epochs (depths)
plt.figure(figsize=(12, 6))

# Plot Accuracy
plt.subplot(1, 2, 1)
plt.plot(range(1, 21), accuracies, marker='o', color='blue')
plt.title('Accuracy vs Epochs (Tree Depth)')
plt.xlabel('Epoch (Tree Depth)')
plt.ylabel('Accuracy')

# Plot Validation Loss
plt.subplot(1, 2, 2)
plt.plot(range(1, 21), losses, marker='o', color='red')
plt.title('Validation Loss vs Epochs (Tree Depth)')
plt.xlabel('Epoch (Tree Depth)')
plt.ylabel('Validation Loss')

# Save the plot to an image file (e.g., 'plot.png')
plt.tight_layout()
plt.savefig('plot.png')

# No plt.show() since we don't want to display the plot, only save it
