import numpy as np

class LogisticReg:
    def __init__(self):
        pass

    def sigmoid(self, z):
        return 1 / (1+ np.exp(-z))
    
    def init_params(self, X, W= 0, B= 0):
        self.weights = np.full(X.shape[1], W, dtype= float) # change to float to avoid problems with type
        self.bias = float(B)
        print(f'Parameters initalized with Weights: {W} and Bias: {B}')

    def log_loss(self, y_pred, Y):
        eps = 1e-15
        y_pred = np.clip(y_pred, eps, 1 - eps) # avoid errors for log(0) and log(1)

        return -np.mean(Y * np.log(y_pred) + (1-Y)*np.log(1- y_pred))
        
    def gradient_descent(self, X, Y, y_pred, loss, lr):
        m = X.shape[0]
        dw = (1/m) * ((y_pred - Y) @ X)
        db = (1/m) * np.sum(y_pred - Y)
        self.weights -= lr* dw
        self.bias -= lr* db
        
    def fit(self, X, Y, lr= 0.01, n_iter= 500):
        for i in range(n_iter): 
            z = X @ self.weights + self.bias # @ is dot product
            y_pred = self.sigmoid(z)
            loss = self.log_loss(y_pred, Y)
            y_pred_class = (y_pred >= 0.5).astype(int)
            accuracy = np.mean(y_pred_class == Y)
            if i % 50 == 0:
                print(f'Iteration: {i} - Loss: {loss:.4f}, Accuracy: {accuracy:.4f}')
        
            self.gradient_descent(X, Y, y_pred, loss, lr)

        print(f'Training Results - Loss: {loss:.4f}, Accuracy: {accuracy:.4f}')

    def predict(self, X, threshold = 0.5):
        z = X @ self.weights + self.bias
        y_pred = self.sigmoid(z)
        return (y_pred >= threshold).astype(int)