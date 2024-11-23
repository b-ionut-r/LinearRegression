import numpy as np




class LinearRegression:

    def __init__(self, lr = 0.001, n_iters = 10000, alpha = 0):
        
        self.lr = lr
        self.n_iters = n_iters
        self.alpha = alpha

        self.weights = None
        self.bias = None
        self.weights_gradient = None
        self.bias_gradient = None

        self.n_samples = None
        self.n_feats = None


        self.is_fitted = False

    def _init_weights(self):
        self.weights = np.random.rand(self.n_feats)
        self.bias = np.random.rand(1)
        self.weights_gradient = np.empty(self.n_feats)

    
    def get_weights(self):
        return self.weights, self.bias


    def fit(self, X, y):

        assert len(X) == len(y), "X and y must have the same number of samples"
        if len(X.shape) > 1:
            self.n_samples, self.n_feats = X.shape
        else:
            self.n_samples = len(X)
            self.n_feats = 1
            X = X[:, np.newaxis]


        self._init_weights()
        self.weights

        for i in range(self.n_iters + 1):

            y_hat = self.predict(X)
            cost = sum((y_hat - y) ** 2)
            if i % 1000 == 0:
                print(f"Iter: {i}. Total Cost: {cost}. MSE: {cost / self.n_samples}")

            
            self.weights_gradient = np.matmul(2 * (y_hat - y), X) / self.n_samples
            self.weights_gradient = self.weights_gradient + self.alpha * (2 * self.weights) # l2 (Ridge) regularization term
            self.bias_gradient = sum(2 * (y_hat - y)) / self.n_samples

            self.weights = self.weights - self.lr * self.weights_gradient
            self.bias = self.bias - self.lr * self.bias_gradient


        self.is_fitted = True

    
    def predict(self, X):
        if len(X.shape) == 1:
            X = X[:, np.newaxis]
        return np.matmul(X, self.weights) + self.bias
    

    def score(self, y_true, y_pred):

        mae = np.mean(np.abs(y_pred - y_true)).item()
        mse = np.mean((y_pred - y_true) ** 2).item()
        rmse = np.sqrt(mse).item()

        return {
            "mae": mae,
            "mse": mse,
            "rmse": rmse
        }