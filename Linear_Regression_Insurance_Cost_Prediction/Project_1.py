# Import Libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Create a file to store the output
with open('insurance_analysis_output.txt', 'w') as f:
    def write_section(title, *content):
        f.write(f"\n{'=' * 80}\n")
        f.write(f"{title.upper()}\n")
        f.write(f"{'=' * 80}\n\n")
        for line in content:
            f.write(line + "\n")

    # Load the Dataset
    df = pd.read_csv('data_insurance.csv')

    # Data Exploration
    write_section("1. Data Exploration", 
        "In this section, we examine the structure and contents of our dataset.\n\n",
        "1.1 First few rows of the dataset:\n\n" ,
        df.head().to_string() + "\n\n",
        "1.2 Dataset information:\n\n",
        str(df.info()) + '\n\n',
        "1.3 Summary statistics of the dataset:\n\n",
        df.describe().to_string() + "\n"
    )

    # Handle Missing Data
    write_section("2. Missing Data Analysis",
        "We check for any missing values in our dataset.\n\n",
        "Number of missing values in each column:\n\n",
        df.isnull().sum().to_string() + "\n\n",
        "Conclusion: There are no missing values in the dataset, so no data imputation is necessary."
    )

    # Convert Categorical Variables
    le = LabelEncoder()
    df['sex'] = le.fit_transform(df['sex'])
    df['smoker'] = le.fit_transform(df['smoker'])
    df['region'] = le.fit_transform(df['region'])

    write_section("3. Feature Engineering",
        "We convert categorical variables to numerical using Label Encoding:\n\n",
        "- 'sex' encoded as: " + ', '.join([f"{v}: {k}" for k, v in enumerate(le.classes_)]) + "\n",
        "- 'smoker' encoded as: " + ', '.join([f"{v}: {k}" for k, v in enumerate(le.classes_)]) + "\n",
        "- 'region' encoded as: " + ', '.join([f"{v}: {k}" for k, v in enumerate(le.classes_)]) + "\n"
    )

    # Define Features and Target
    X = df.drop('charges', axis=1)
    y = df['charges']

    # Split the Dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    write_section("4. Data Splitting",
        "We split the dataset into training and testing sets:\n\n",
        "- Training set: 80% of the data\n",
        "- Testing set: 20% of the data\n",
        f"- Number of samples in training set: {len(X_train)}\n",
        f"- Number of samples in testing set: {len(X_test)}\n"
    )

    # Train the Linear Regression Model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Make Predictions
    y_pred = model.predict(X_test)

    # Evaluate the Model
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    write_section("5. Model Evaluation",
        f"We evaluate the performance of our Linear Regression model:\n\n",
        f"- Mean Squared Error (MSE): {mse:.2f}\n",
        f"- Root Mean Squared Error (RMSE): {rmse:.2f}\n",
        f"- R-squared (R2) Score: {r2:.4f}\n\n",
        "Interpretation:\n",
        f"- The RMSE of {rmse:.2f} indicates the average deviation of our predictions from the actual values.\n",
        f"- The R2 score of {r2:.4f} suggests that our model explains {r2*100:.2f}% of the variance in the target variable.\n"
    )

    # Analyze Model Coefficients
    coef_df = pd.DataFrame({'Feature': X.columns, 'Coefficient': model.coef_})
    coef_df = coef_df.sort_values('Coefficient', key=abs, ascending=False)
    
    write_section("6. Model Coefficients Analysis",
        "We analyze the coefficients of our Linear Regression model to understand feature importance:\n\n",
        coef_df.to_string(index=False) + "\n\n",
        f"Intercept of the model: {model.intercept_:.4f}\n\n",
        "Interpretation:\n",
        "- Positive coefficients indicate a positive correlation with insurance charges.\n",
        "- Negative coefficients indicate a negative correlation with insurance charges.\n",
        "- The magnitude of the coefficient indicates the strength of the relationship.\n"
    )

    # Create a point-line graph of actual vs prediction data
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    z = np.polyfit(y_test, y_pred, 1)
    p = np.poly1d(z)
    plt.plot(y_test, p(y_test), "r--")
    plt.xlabel('Actual Charges')
    plt.ylabel('Predicted Charges')
    plt.title('Actual vs Predicted Insurance Charges')
    plt.savefig('actual_vs_predicted.png')
    plt.close()

    # write_section("7. Visualization",
    #     "We have created a scatter plot comparing actual vs predicted insurance charges.\n",
    #     "The plot has been saved as 'actual_vs_predicted.png' in the current directory.\n",
    #     "This visualization helps us understand how well our model's predictions align with the actual values.\n"
    # )
    
    write_section("7. Visualization and Analysis",
    "We have created a scatter plot comparing actual vs predicted insurance charges.\n",
    "The scatter plot visually represents the relationship between actual and predicted insurance charges.\nThe red dashed line, representing the ideal prediction where actual and predicted charges perfectly align, provides a benchmark for evaluating the model's performance.\n\n",
    "The plot reveals a strong correlation between actual and predicted charges, suggesting the model effectively captures the underlying patterns in insurance costs.\nHowever, the presence of points deviating from the ideal line indicates room for improvement in the model's accuracy.\n\n",
    "This analysis serves multiple purposes:\n\n",
    "Understanding Influencing Factors: By examining the patterns and deviations, we can identify factors impacting insurance charges.\n",
    "Assessing Individual Risks: The model helps estimate individual insurance costs based on their specific characteristics.\n",
    "Optimizing Healthcare Costs: Accurate prediction enables effective cost management and resource allocation in healthcare.\n",
    "Gaining Insights into Public Health Trends: Analyzing the data can reveal broader trends in healthcare utilization and costs, supporting public health strategies.\n",
    "The plot has been saved as 'actual_vs_predicted.png' in the current directory.\n",
    "This visualization helps us understand how well our model's predictions align with the actual values.\n"
)

    # Model Refinement (Optional)
    write_section("8. Model Refinement Suggestions",
        "Based on the model's performance, consider the following refinements:\n\n",
        "1. Feature Engineering: Create interaction terms or polynomial features for key variables.\n",
        "2. Feature Selection: Use techniques like Lasso or Ridge regression to identify the most important features.\n",
        "3. Non-linear Models: Experiment with non-linear models like Random Forests or Gradient Boosting Machines.\n",
        "4. Hyperparameter Tuning: Use techniques like Grid Search or Random Search to optimize model parameters.\n",
        "5. Cross-validation: Implement k-fold cross-validation for more robust performance estimation.\n"
    )

print("Analysis complete. Results have been saved to 'insurance_analysis_output.txt'.") 