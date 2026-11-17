
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Load the dataset
homes = pd.read_csv("real_estate.csv")

# Convert distance into units of 100 meters
homes["distance_100m"] = (
    homes["X3 distance to the nearest MRT station"] / 100
)

# Select input and target variables
X = homes[["distance_100m"]]
y = homes["Y house price of unit area"]

# Create and train Linear Regression model
model = LinearRegression()
model.fit(X, y)


# Task 1(a): Fit Linear Regression
print("\n========== TASK 1(a) ==========")
print("Linear Regression Model")
print("Slope:", model.coef_[0])
print("Intercept:", model.intercept_)


# Task 1(b): Predict House Price at 500 meters
print("\n========== TASK 1(b) ==========")

new_house = pd.DataFrame({
    "distance_100m": [5]
})

predicted_price = model.predict(new_house)

print("Distance from Metro Station: 500 meters")
print("Predicted House Price:", predicted_price[0])


# Task 1(c): Explain Price Change
print("\n========== TASK 1(c) ==========")

print("Slope Explanation:")
print(
    f"For every additional 100 meters from the metro station, "
    f"the predicted house price decreases by "
    f"{abs(model.coef_[0]):.2f} price units."
)


# Plot observations and fitted regression line
plt.figure(figsize=(9, 6))

plt.scatter(
    X["distance_100m"],
    y,
    color="blue",
    label="Actual Housing Prices"
)

plt.plot(
    X["distance_100m"],
    model.predict(X),
    color="red",
    label="Fitted Regression Line"
)

plt.xlabel("Distance from MRT Station (100 meters)")
plt.ylabel("House Price per Unit Area")
plt.title("Housing Prices vs Distance from Metro Station")

plt.legend()
plt.grid(True)

# Save the graph
plt.savefig("task1_housing.png")
plt.show()

print("\nGraph saved successfully as task1_housing.png")
