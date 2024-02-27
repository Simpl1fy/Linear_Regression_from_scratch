import numpy as np
from LinearRegression import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import datasets

X,y = datasets.make_regression(n_samples=100, n_features=1, noise=20, random_state=4)



X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=0.3)

reg = LinearRegression(learning_rate=0.01)
reg.fit(X_train, y_train)
predictions = reg.predict(X_test)

# Function to calculate MSE
def mse(ypred, y_test):
    return np.mean((ypred - y_test) ** 2)

print(f"Mean Squared Error is: {mse(predictions, y_test)}")