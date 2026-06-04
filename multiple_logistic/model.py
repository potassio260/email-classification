# Imports
import numpy as np
import math
from binary_logistic.model import BinaryLogisticRegression

# Multiple Logistic Regression Class
class MultipleLogisticRegression:
    def __init__(self, learning_rate=0.01, number_of_iterations=1000):
        self.learning_rate = learning_rate
        self.number_of_iterations = number_of_iterations
        self.models = []
        self.classes = []
        
    def fit(self, X, y):
        # Handle sparse matrices from TfidfVectorizer
        if hasattr(X, 'toarray'):
            X = X.toarray()
        
        X = np.array(X)
        y = np.array(y)
        
        self.classes = np.unique(y)
        for y_label in self.classes:
            binary_y = (y == y_label).astype(int)
            binary_model = BinaryLogisticRegression(
                learning_rate=self.learning_rate,
                number_of_iterations=self.number_of_iterations
            )
            binary_model.fit(X, binary_y)
            self.models.append(binary_model)
            
            print(f"Training complete for class {y_label}.")

    def predict(self, X):
        # Handle sparse matrices from TfidfVectorizer
        if hasattr(X, 'toarray'):
            X = X.toarray()
        
        X = np.array(X)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        
        # Get probabilities from each binary classifier (after sigmoid)
        # Use predict() instead of z_predict() to get sigmoid outputs
        probs_raw = np.column_stack([model.predict(X) for model in self.models])
        
        # Normalize probabilities across classes (they should sum to 1)
        probs = probs_raw / np.sum(probs_raw, axis=1, keepdims=True)
        
        # If only one sample, return as a 1D list instead of [[...]]
        if probs.shape[0] == 1:
            return probs[0].tolist()
        else:
            return probs.tolist()