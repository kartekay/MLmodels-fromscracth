import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
class BostonDataset:
    def __init__(self):
        self.url = "http://lib.stat.cmu.edu/datasets/boston"
        self.feature_names = ["CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE", "DIS", "RAD", "TAX", "PTRATIO", "B",
                              "LSTAT"]
    def datasetload(self):

        df0 = pd.read_csv(self.url, sep="\s+", skiprows=22, header=None)
        data = np.hstack([df0.values[::2, :], df0.values[1::2, :2]])
        target = df0.values[1::2, 2]
        dataset = {
            'data': data,
            'target': target,
            'feature_names': self.feature_names,
            'DESCR': 'Boston House Prices dataset'
        }

        return dataset
bostondataset = BostonDataset().datasetload()
boston = pd.DataFrame(bostondataset['data'], columns=bostondataset['feature_names'])
boston['MEDV'] = bostondataset['target']
boston = boston.dropna()
X = boston[bostondataset['feature_names']]
y = boston['MEDV']
mask = (np.abs(X - X.mean()) / X.std() < 3).all(axis=1)
X = X[mask]
y = y[mask]
scaler = StandardScaler()
X = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

class MSECustom:
    def __init__(self, lr=0.00001, n_iters=1000):
        self.lr = lr
        self.n_iters = n_iters
        self.w = None
        self.b = None
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)
        self.b = 0

        for _ in range(self.n_iters):

            y_pred = np.dot(X, self.w) + self.b


            dw = (1 / n_samples) * np.dot(X.T, (y_pred - y))
            db = (1 / n_samples) * np.sum(y_pred - y)


            self.w -= self.lr * dw
            self.b -= self.lr * db
    def predict(self, X):
        return np.dot(X, self.w) + self.b



modelmse = MSECustom(lr=0.00001, n_iters=1000)
modelmse.fit(X_train, y_train)
y_pred_mse = modelmse.predict(X_test)
mse_scratch_mse = mean_squared_error(y_test, y_pred_mse)
mae_scratch_mse = mean_absolute_error(y_test, y_pred_mse)
r2_scratch_mse = r2_score(y_test, y_pred_mse)
explained_variance_scratch_mse = 1 - (np.var(y_test - y_pred_mse) / np.var(y_test))


print("\nCustom MSE Model Results:")
print(
    f"  MSE: {mse_scratch_mse:.4f}, MAE: {mae_scratch_mse:.4f}, R²: {r2_scratch_mse:.4f}, Explained Variance: {explained_variance_scratch_mse:.4f}")

class MAECustom:
    def __init__(self, lr=0.0001, n_iters=1000):
        self.lr = lr
        self.n_iters = n_iters
        self.w = None
        self.b = None
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)
        self.b = 0

        for _ in range(self.n_iters):

            y_pred = np.dot(X, self.w) + self.b


            dw = np.sign(y_pred - y).dot(X) / n_samples
            db = np.sum(np.sign(y_pred - y)) / n_samples


            self.w -= self.lr * dw
            self.b -= self.lr * db
    def predict(self, X):
        return np.dot(X, self.w) + self.b

modelmae = MAECustom(lr=0.0001, n_iters=1000)
modelmae.fit(X_train, y_train)
y_pred_mae = modelmae.predict(X_test)
mse_scratch_mae = mean_squared_error(y_test, y_pred_mae)
mae_scratch_mae = mean_absolute_error(y_test, y_pred_mae)
r2_scratch_mae = r2_score(y_test, y_pred_mae)
explained_variance_scratch_mae = 1 - (np.var(y_test - y_pred_mae) / np.var(y_test))
print("\nCustom MAE Model Results:")
print(f"  MSE: {mse_scratch_mae:.4f}, MAE: {mae_scratch_mae:.4f}, R²: {r2_scratch_mae:.4f}, Explained Variance: {explained_variance_scratch_mae:.4f}")