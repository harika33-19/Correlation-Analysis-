# Correlation-Analysis-
# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")

# Create Dataset
data = {
    'Study_Hours': [2, 3, 4, 5, 6, 7, 8, 9, 1, 10],
    'Sleep_Hours': [8, 7, 7, 6, 8, 6, 7, 5, 9, 6],
    'Attendance': [60, 65, 70, 75, 80, 85, 90, 95, 50, 98],
    'Marks': [45, 50, 55, 65, 70, 75, 85, 90, 40, 95]
}

df = pd.DataFrame(data)

# Display Dataset
print("Dataset:")
print(df)

# Explore Dataset
print("\nDataset Information:")
print(df.info())

print("\nStatistical Summary:")
print(df.describe())

print("\nMissing Values:")
print(df.isnull().sum())

# Calculate Correlation Matrix
correlation_matrix = df.corr()

print("\nCorrelation Matrix:")
print(correlation_matrix)

# Correlation Heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(
    correlation_matrix,
    annot=True,
    cmap='coolwarm',
    fmt='.2f',
    linewidths=0.5
)
plt.title("Correlation Heatmap")
plt.show()

# Visualization 1: Study Hours vs Marks
plt.figure(figsize=(7, 5))
sns.scatterplot(data=df, x='Study_Hours', y='Marks')
plt.title("Study Hours vs Marks")
plt.show()

# Visualization 2: Attendance vs Marks
plt.figure(figsize=(7, 5))
sns.scatterplot(
    data=df,
    x='Attendance',
    y='Marks',
    color='green'
)
plt.title("Attendance vs Marks")
plt.show()

# Visualization 3: Sleep Hours vs Marks
plt.figure(figsize=(7, 5))
sns.scatterplot(
    data=df,
    x='Sleep_Hours',
    y='Marks',
    color='orange'
)
plt.title("Sleep Hours vs Marks")
plt.show()

# Identify Strong Positive Correlations
print("\nStrong Positive Correlations:")
print(
    correlation_matrix[
        (correlation_matrix > 0.7) &
        (correlation_matrix < 1.0)
    ].stack()
)

# Identify Strong Negative Correlations
print("\nStrong Negative Correlations:")
print(
    correlation_matrix[
        correlation_matrix < -0.7
    ].stack()
)

print("\nCorrelation Analysis Completed Successfully!")
