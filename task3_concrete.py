
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import train_test_split, KFold
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error


# Load the Concrete Strength dataset
concrete = pd.read_csv("concrete_strength.csv")

# Remove duplicate rows
concrete = concrete.drop_duplicates()

# Select input and target variables
target = "Concrete compressive strength"

X = concrete.drop(columns=target)
y = concrete[target]


# Task 3(a): Split Data and Linear Regression Baseline
print("\n========== TASK 3(a) ==========")

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

print("Training rows:", len(X_train))
print("Testing rows:", len(X_test))

# Create five-fold cross-validation
folds = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

scoring = "neg_root_mean_squared_error"

# Create Linear Regression baseline
linear_model = LinearRegression()

# Calculate baseline cross-validation RMSE
baseline_rmse = -cross_val_score(
    linear_model,
    X_train,
    y_train,
    cv=folds,
    scoring=scoring
)

baseline_mean = baseline_rmse.mean()

print("Linear Regression Fold RMSEs:", baseline_rmse.round(2))
print("Baseline Mean CV RMSE:", round(baseline_mean, 2), "MPa")


# Task 3(b): Tune Polynomial Ridge Regression
print("\n========== TASK 3(b) ==========")

# Create Polynomial Ridge Pipeline
ridge_model = Pipeline([
    ("poly", PolynomialFeatures(include_bias=False)),
    ("scale", StandardScaler()),
    ("ridge", Ridge())
])

# Define parameter grid
param_grid = {
    "poly__degree": [1, 2, 3],
    "ridge__alpha": [0.01, 0.1, 1, 10, 100]
}

# Perform Grid Search
search = GridSearchCV(
    ridge_model,
    param_grid,
    cv=folds,
    scoring=scoring,
    refit=True
)

search.fit(X_train, y_train)

print("Best Parameters:", search.best_params_)
print("Best Mean CV RMSE:", round(-search.best_score_, 2), "MPa")

# Display three highest-ranked settings
results = pd.DataFrame(search.cv_results_)

top_three = results.sort_values(
    "rank_test_score"
).head(3)

print("\nTop 3 Grid Search Results:")

for _, row in top_three.iterrows():
    print(
        "Degree:", row["param_poly__degree"],
        "| Alpha:", row["param_ridge__alpha"],
        "| Mean CV RMSE:", round(-row["mean_test_score"], 2),
        "| Rank:", row["rank_test_score"]
    )


# Task 3(c): Evaluate the Selected Model on Test Data
print("\n========== TASK 3(c) ==========")

# Predict using the best refitted model
test_predictions = search.predict(X_test)

# Calculate test RMSE and MAE
test_rmse = np.sqrt(
    mean_squared_error(y_test, test_predictions)
)

test_mae = mean_absolute_error(
    y_test, test_predictions
)

print("Test RMSE:", round(test_rmse, 2), "MPa")
print("Test MAE:", round(test_mae, 2), "MPa")

# Plot actual vs predicted values
plt.figure(figsize=(9, 6))

plt.scatter(
    y_test,
    test_predictions,
    label="Predicted Values"
)

# Reference line y = x
minimum = min(y_test.min(), test_predictions.min())
maximum = max(y_test.max(), test_predictions.max())

plt.plot(
    [minimum, maximum],
    [minimum, maximum],
    color="red",
    label="Reference Line (y = x)"
)

plt.xlabel("Actual Concrete Strength (MPa)")
plt.ylabel("Predicted Concrete Strength (MPa)")
plt.title("Actual vs Predicted Concrete Strength")

plt.legend()
plt.grid(True)

# Save graph
plt.savefig("task3_concrete.png")
plt.show()

print("Graph saved successfully as task3_concrete.png")


# Task 3(d): Model Comparison and Explanation
print("\n========== TASK 3(d) ==========")

print("Baseline Mean CV RMSE:", round(baseline_mean, 2), "MPa")
print("Tuned Mean CV RMSE:", round(-search.best_score_, 2), "MPa")

if -search.best_score_ < baseline_mean:
    print("Tuned Ridge Model performs better than Linear Regression.")
else:
    print("Linear Regression performs better.")

print("\nInteraction Term Example:")
print("Cement x Water")

print("\nScaling Explanation:")
print("Scaling puts features on a similar scale.")
print("It prevents large-valued features from dominating Ridge regularization.")
