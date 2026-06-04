# Imports
import numpy as np

# Binary Logistic Regression Class
class BinaryLogisticRegression:
    def __init__(self, learning_rate=0.01, number_of_iterations=1000):
        self.learning_rate = learning_rate
        self.number_of_iterations = number_of_iterations
        self.weights = None
        self.bias = None
        
    def _sigmoid(self, z:int):
        return 1 / (1 + np.exp(-z))
    
    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)
        num_samples, num_features = X.shape
        self.weights = np.zeros(num_features)
        self.bias = 0
        
        for _ in range(self.number_of_iterations):
            linear_model = np.dot(X, self.weights) + self.bias
            y_predicted = self._sigmoid(linear_model)
            
            dw = (1 / num_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / num_samples) * np.sum(y_predicted - y)
            
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            
    def predict(self, X):
        X = np.array(X)
        linear_model = np.dot(X, self.weights) + self.bias
        y_predicted = self._sigmoid(linear_model)
        return y_predicted
    
    def z_predict(self, X):
        X = np.array(X)
        linear_model = np.dot(X, self.weights) + self.bias
        z = linear_model
        return z
    
    def loss(self, X, y):
        X = np.array(X)
        y = np.array(y)
        num_samples = X.shape[0]
        linear_model = np.dot(X, self.weights) + self.bias
        y_predicted = self._sigmoid(linear_model)
        
        y_predicted = np.clip(y_predicted, 1e-10, 1 - 1e-10)
        
        # calculate binary cross-entropy loss
        loss = -(1 / num_samples) * np.sum(y * np.log(y_predicted) + (1 - y) * np.log(1 - y_predicted))
        return loss
    