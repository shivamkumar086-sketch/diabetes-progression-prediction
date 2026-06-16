import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# =====================================
# Create Output Folder
# =====================================
os.makedirs("output", exist_ok=True)

# =====================================
# Load Dataset
# =====================================
print("Loading Diabetes Dataset...")

diabetes = load_diabetes()

df = pd.DataFrame(
    diabetes.data,
    columns=diabetes.feature_names
)

df["disease_progression"] = diabetes.target

# =====================================
# Basic Information
# =====================================
print("\nFirst 5 Rows:")
print(df.head())

print("\nDataset Shape:")
print(df.shape)

print("\nColumn Names:")
print(df.columns.tolist())

print("\nData Types:")
print(df.dtypes)

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDuplicate Rows:")
print(df.duplicated().sum())

# =====================================
# Descriptive Statistics
# =====================================
print("\nDescriptive Statistics:")
print(df.describe())

# =====================================
# Histograms
# =====================================
df.hist(figsize=(15, 12))
plt.suptitle("Feature Distributions")
plt.tight_layout()
plt.savefig("output/feature_distributions.png")
plt.show()

# =====================================
# Correlation Heatmap
# =====================================
corr = df.corr()

plt.figure(figsize=(12, 10))
sns.heatmap(
    corr,
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Correlation Heatmap")
plt.savefig("output/correlation_heatmap.png")
plt.show()

# =====================================
# Top Correlations with Target
# =====================================
target_corr = (
    corr["disease_progression"]
    .sort_values(ascending=False)
)

print("\nCorrelation with Disease Progression:")
print(target_corr)

# =====================================
# Scatter Plot:
# BMI vs Disease Progression
# =====================================
plt.figure(figsize=(8, 5))

sns.scatterplot(
    x=df["bmi"],
    y=df["disease_progression"]
)

plt.title("BMI vs Disease Progression")
plt.xlabel("BMI")
plt.ylabel("Disease Progression")

plt.savefig("output/bmi_vs_progression.png")
plt.show()

# =====================================
# Machine Learning
# =====================================
X = df.drop("disease_progression", axis=1)
y = df["disease_progression"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# =====================================
# Train Model
# =====================================
model = LinearRegression()

model.fit(X_train, y_train)

# =====================================
# Predictions
# =====================================
y_pred = model.predict(X_test)

# =====================================
# Evaluation Metrics
# =====================================
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\n========== MODEL PERFORMANCE ==========")

print(f"Mean Absolute Error: {mae:.2f}")
print(f"Mean Squared Error: {mse:.2f}")
print(f"Root Mean Squared Error: {rmse:.2f}")
print(f"R² Score: {r2:.4f}")

# =====================================
# Actual vs Predicted
# =====================================
plt.figure(figsize=(8, 5))

plt.scatter(y_test, y_pred)

plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Actual vs Predicted")

plt.savefig("output/actual_vs_predicted.png")
plt.show()

# =====================================
# Feature Importance
# =====================================
importance = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": model.coef_
})

importance["Absolute"] = (
    importance["Coefficient"]
    .abs()
)

importance = importance.sort_values(
    by="Absolute",
    ascending=False
)

print("\nFeature Importance:")
print(importance[["Feature", "Coefficient"]])

plt.figure(figsize=(10, 6))

sns.barplot(
    x="Coefficient",
    y="Feature",
    data=importance
)

plt.title("Feature Importance")

plt.savefig("output/feature_importance.png")
plt.show()

# =====================================
# Key Insights
# =====================================
print("\n========== KEY INSIGHTS ==========")

print(
    "\nFeatures most associated with disease progression:"
)

print(
    target_corr.head()
)

print(
    "\nThe model explains "
    f"{r2*100:.2f}% of the variation "
    "in disease progression."
)

print(
    "\nHigher BMI values tend to be associated "
    "with greater disease progression."
)

print("\nProject Completed Successfully!")

print(
    "\nGraphs have been saved "
    "inside the 'output' folder."
)