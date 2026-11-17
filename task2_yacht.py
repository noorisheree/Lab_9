
import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures


# Load the Yacht Hydrodynamics dataset
yachts = pd.read_csv(
    "yacht_hydrodynamics.data",
    sep=r"\s+",
    header=None
)

# Select input and target variables
X = yachts[[5]]
y = yachts[6]


# Task 2(a): Build Linear and Quadratic Models
print("\n========== TASK 2(a) ==========")

# Linear Regression Model
linear_model = LinearRegression()

# Quadratic Polynomial Regression Model
quadratic_model = Pipeline([
    ("poly", PolynomialFeatures(degree=2, include_bias=False)),
    ("linear", LinearRegression())
])

print("Linear Regression Model created successfully.")
print("Quadratic Polynomial Regression Model created successfully.")


# Task 2(b): Five-Fold Cross-Validation
print("\n========== TASK 2(b) ==========")

# Create five shuffled folds
folds = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

# Use negative RMSE scoring
scoring = "neg_root_mean_squared_error"

# Calculate Linear Regression RMSE
linear_rmse = -cross_val_score(
    linear_model,
    X,
    y,
    cv=folds,
    scoring=scoring
)

# Calculate Quadratic Regression RMSE
quadratic_rmse = -cross_val_score(
    quadratic_model,
    X,
    y,
    cv=folds,
    scoring=scoring
)

# Display fold results
print("\nLinear Regression Fold RMSEs:")
print(linear_rmse.round(2))

print("Linear Regression Mean RMSE:")
print(round(linear_rmse.mean(), 2))

print("\nQuadratic Regression Fold RMSEs:")
print(quadratic_rmse.round(2))

print("Quadratic Regression Mean RMSE:")
print(round(quadratic_rmse.mean(), 2))


# Task 2(c): Compare and Select the Better Model
print("\n========== TASK 2(c) ==========")

if linear_rmse.mean() < quadratic_rmse.mean():
    print("Better Model: Linear Regression")
    print("Linear Regression has lower mean RMSE.")
else:
    print("Better Model: Quadratic Polynomial Regression")
    print("Quadratic Regression has lower mean RMSE.")

print("\nModel Comparison:")
print("Linear Mean RMSE:", round(linear_rmse.mean(), 2))
print("Quadratic Mean RMSE:", round(quadratic_rmse.mean(), 2))