import numpy as np
from LinearRegression import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt

X,y = datasets.make_regression(n_samples=100, n_features=1, noise=20, random_state=4)



X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=0.3)

reg = LinearRegression(learning_rate=0.01)
reg.fit(X_train, y_train)
predictions = reg.predict(X_test)

# Function to calculate MSE
def mse(ypred, y_test):
    return np.mean((ypred - y_test) ** 2)

print(f"Mean Squared Error is: {mse(predictions, y_test)}")

y_pred_line = reg.predict(X)
cmap = plt.get_cmap('viridis')
fig = plt.figure(figsize=(8,6))
m1 = plt.scatter(X_train, y_train, color=cmap(0.9), s=10)
m2 = plt.scatter(X_test, y_test, color=cmap(0.5), s=10)
plt.plot(X, y_pred_line, color='black', linewidth=2, label='Prediction')
plt.show()