import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

X = np.array([[75, 85], [60, 70], [90, 95], [80, 75], [70, 60], [55, 65]])
Y = np.array([1, 0, 1, 1, 0, 0])  # 1 for "clever," 0 for "average"
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
model = LogisticRegression()
model.fit(X_train, Y_train)

# Make predictions on the test data
Y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(Y_test, Y_pred)

print(f"Accuracy: {accuracy * 100:.2f}%")

# Visualize the data points and decision boundary
plt.figure(figsize=(8, 6))

# Plot the actual data points
plt.scatter(X[Y == 0][:, 0], X[Y == 0][:, 1], label="Average")
plt.scatter(X[Y == 1][:, 0], X[Y == 1][:, 1], label="Clever")

# Create a mesh grid to plot the decision boundary
xx, yy = np.meshgrid(np.linspace(50, 100, 100), np.linspace(50, 100, 100))
Z = model.predict(np.c_[xx.ravel(), yy.ravel()])

# Plot the decision boundary
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, cmap=plt.cm.RdBu, alpha=0.6)

plt.xlabel("Exam 1 Score")
plt.ylabel("Exam 2 Score")
plt.legend()
plt.title("Logistic Regression: Student Classification")
plt.show()
