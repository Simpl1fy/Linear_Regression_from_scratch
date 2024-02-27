import numpy as np

class LinearRegression:

    def __init__(self, learning_rate = 0.001, n_iters = 1000):    # Function to initialize the parameters of the model
        self.learning_rate = learning_rate
        self.n_iters = n_iters
        self.weights = None
        self.bias = None
    
    def fit(self, X, y):
        
        # Calculating the shape the data
        X_samples, X_features = X.shape

        # Creating a array of 0's for weights
        self.weights = np.zeros(X_features)
        # Setting bias to 0
        self.bias = 0

        # Gradient Descent function
        for _ in range(self.n_iters):
            # Predicting
            y_pred = np.dot(X_samples, self.weights) + self.bias

            # Calculating the partial derivatives
            dw = (1/X_samples) * np.dot(X, (y_pred - y))
            db = (1/X_samples) + np.sum(y_pred - y)

            # Updating the parameters, to find the global minima
            self.weights = self.weights - (self.learning_rate * dw)
            self.bias = self.bias - (self.learning_rate * db)
    

    # Predicting on given data
    def predict(self, X):
        y_pred = np.dot(X, self.weights) + self.bias
        return y_pred   # Returning output