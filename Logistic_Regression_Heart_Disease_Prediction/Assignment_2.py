# Importing necessary libraries
import pandas as pd

import matplotlib.pyplot as plt  # Importing matplotlib for plotting
import seaborn as sns  # For heat map
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.impute import SimpleImputer  # To handle missing values
from sklearn.preprocessing import StandardScaler  # To scale data

# Load the dataset from CSV file (replace 'framingham.csv' with the actual filename)
df = pd.read_csv('framingham.csv')

# Check for missing values and log the result into the output file
with open('output.txt', 'w') as f:
    f.write("********** Heart Study - Logistic Regression Results **********\n\n")
    f.write("### Missing values in each column ###\n")
    f.write(str(df.isnull().sum()) + "\n\n")
    f.write("-----------------------------------------------------------------\n\n")

# Impute missing values with median for numerical columns
imputer = SimpleImputer(strategy='median')
df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

# Features and Target variable
X = df_imputed[['male', 'age', 'education', 'currentSmoker', 'cigsPerDay', 'BPMeds', 'prevalentStroke',
                'prevalentHyp', 'diabetes', 'totChol', 'sysBP', 'diaBP', 'BMI', 'heartRate', 'glucose']]

y = df_imputed['TenYearCHD']  # Target variable indicating the 10-year CHD risk

# Scale the features to standardize them (mean=0, variance=1)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Splitting the dataset into training (70%) and testing (30%) sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# Initialize the Logistic Regression model with a higher number of iterations
log_reg = LogisticRegression(max_iter=5000)  # Increase max_iter to 5000

# Train the model on the training data
log_reg.fit(X_train, y_train)

# Predict the target on the test data
y_pred = log_reg.predict(X_test)

# Model Evaluation
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

# Optional: Checking model coefficients to understand the importance of each feature
coeff_df = pd.DataFrame(log_reg.coef_[0], X.columns, columns=['Coefficient'])

# Write all the results into the output file
with open('output.txt', 'a') as f:
    f.write("********** Model Evaluation Results **********\n\n")
    f.write(f"### Model Accuracy ###\nAccuracy: {accuracy:.4f}\n\n")
    f.write("-----------------------------------------------------------------\n\n")
    
    f.write("### Confusion Matrix ###\n")
    f.write(str(conf_matrix) + "\n\n")
    f.write("-----------------------------------------------------------------\n\n")
    
    f.write("### Classification Report ###\n")
    f.write(class_report + "\n")
    f.write("-----------------------------------------------------------------\n\n")
    
    f.write("### Model Coefficients ###\n")
    f.write(coeff_df.to_string() + "\n")
    f.write("-----------------------------------------------------------------\n\n")

# Saving the model predictions and evaluation metrics for further analysis
results_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
results_df.to_csv('logistic_regression_results.csv', index=False)

# Plotting Confusion Matrix as a Heat Map
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=["No CHD", "CHD"], yticklabels=["No CHD", "CHD"])
plt.title('Confusion Matrix Heatmap')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()

# Save the heatmap as an image file
plt.savefig('confusion_matrix_heatmap.png')

# Plotting the Actual vs Predicted values as a point-line graph
plt.figure(figsize=(10, 6))
plt.plot(range(len(y_test)), y_test, label='Actual', marker='o', linestyle='-', color='blue')  # Actual values
plt.plot(range(len(y_pred)), y_pred, label='Predicted', marker='x', linestyle='--', color='red')  # Predicted values
plt.title('Actual vs Predicted - Logistic Regression')
plt.xlabel('Sample Index')
plt.ylabel('CHD Risk (0 = No, 1 = Yes)')
plt.legend()
plt.grid(True)
plt.tight_layout()

# Save the plot as an image file
plt.savefig('actual_vs_predicted.png')
print("All information present in the output.txt, and some graph also created.")
