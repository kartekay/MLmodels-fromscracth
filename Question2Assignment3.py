import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

def sigmoid(z):
    z = np.clip(z, -500, 500)  
    return 1 / (1 + np.exp(-z))

def hypothesis(X, theta):
    return sigmoid(np.dot(X, theta))

def compute_loss(X, y, theta, lambda_2):
    m = len(y)
    h = hypothesis(X, theta)
    loss = -(1/m) * (np.dot(y, np.log(h)) + np.dot(1 - y, np.log(1 - h)))
    reg_term_l2 = (lambda_2 / (2 * m)) * np.sum(theta[1:]**2)
    return loss + reg_term_l2

def gradient_descent(X, y, theta, alpha, lambda_2, iterations):
    m = len(y)
    for _ in range(iterations):
        h = hypothesis(X, theta)
        gradient = (1/m) * np.dot(X.T, (h - y))
        l2_term = lambda_2 * theta
        gradient[1:] += l2_term[1:]
        theta -= alpha * gradient
    return theta

def train_logistic_regression(X, y, alpha=0.0001, lambda_2=0, iterations=1000):
    X = np.hstack((np.ones((X.shape[0], 1)), X))
    theta = np.zeros(X.shape[1])
    theta = gradient_descent(X, y, theta, alpha, lambda_2, iterations)
    return theta

def predict(X, theta):
    X = np.hstack((np.ones((X.shape[0], 1)), X))
    return (hypothesis(X, theta) >= 0.5).astype(int)

def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)

def cross_validation(X, y, k, alpha, lambda_2, iterations):
    fold_size = len(X) // k
    accuracies = []
    for i in range(k):
        start = i * fold_size
        end = (i + 1) * fold_size if i != k - 1 else len(X)
        
        X_train = np.concatenate([X[:start], X[end:]], axis=0)
        y_train = np.concatenate([y[:start], y[end:]], axis=0)
        X_val = X[start:end]
        y_val = y[start:end]

        model = LogisticRegressionModel(alpha, lambda_2, iterations)
        model.fit(X_train, y_train)
        y_pred_val = model.predict(X_val)
        accuracies.append(accuracy(y_val, y_pred_val))
    
    return np.mean(accuracies)

def load_data():
    df = pd.read_csv("bank-full.csv", delimiter=";")
    df = df.dropna()
    df = pd.get_dummies(df, drop_first=True)
    X = df.drop('y_yes', axis=1).values
    y = df['y_yes'].values
    return X, y

def apply_pca(X, n_components=2):
    mean = np.mean(X, axis=0)
    X_centered = X - mean
    covariance_matrix = np.cov(X_centered.T)
    eigvals, eigvecs = np.linalg.eig(covariance_matrix)
    sorted_idx = np.argsort(eigvals)[::-1]
    eigvecs_sorted = eigvecs[:, sorted_idx]
    return np.dot(X_centered, eigvecs_sorted[:, :n_components])

def add_polynomial_features(X, degree):
    m = X.shape[0]
    X_poly = X
    for d in range(2, degree + 1):
        for i in range(X.shape[1]):
            X_poly = np.hstack((X_poly, (X[:, i:i+1] ** d)))
    return X_poly

class LogisticRegressionModel:
    def __init__(self, alpha=0.0001, lambda_2=0, iterations=1000):
        self.alpha = alpha
        self.lambda_2 = lambda_2
        self.iterations = iterations
        self.theta = None

    def fit(self, X, y):
        self.theta = train_logistic_regression(X, y, alpha=self.alpha, lambda_2=self.lambda_2, iterations=self.iterations)
        return self

    def predict(self, X):
        return predict(X, self.theta)

X, y = load_data()
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

model_no_reg = LogisticRegressionModel(alpha=0.00005, lambda_2=0, iterations=500)
model_no_reg.fit(X_train, y_train)
y_train_pred_no_reg = model_no_reg.predict(X_train)
y_val_pred_no_reg = model_no_reg.predict(X_val)

print("No regularization:")
print("Train accuracy:", accuracy(y_train, y_train_pred_no_reg))
print("Validation accuracy:", accuracy(y_val, y_val_pred_no_reg))

for degree in range(1, 4):
    X_train_poly = add_polynomial_features(X_train, degree)
    X_val_poly = add_polynomial_features(X_val, degree)
    
    model_poly_no_reg = LogisticRegressionModel(alpha=0.00005, lambda_2=0, iterations=500)
    model_poly_no_reg.fit(X_train_poly, y_train)
    y_train_pred_poly_no_reg = model_poly_no_reg.predict(X_train_poly)
    y_val_pred_poly_no_reg = model_poly_no_reg.predict(X_val_poly)
    
    print(f"\nPolynomial degree {degree} without regularization:")
    print("Train accuracy:", accuracy(y_train, y_train_pred_poly_no_reg))
    print("Validation accuracy:", accuracy(y_val, y_val_pred_poly_no_reg))


X_train_pca = apply_pca(X_train, n_components=5)
X_val_pca = apply_pca(X_val, n_components=5)

model_l2_reg = LogisticRegressionModel(alpha=0.00005, lambda_2=1000, iterations=500)
model_l2_reg.fit(X_train_pca, y_train)
y_train_pred_l2 = model_l2_reg.predict(X_train_pca)
y_val_pred_l2 = model_l2_reg.predict(X_val_pca)

print("\nWith L2 regularization (lambda=1000):")
print("Train accuracy:", accuracy(y_train, y_train_pred_l2))
print("Validation accuracy:", accuracy(y_val, y_val_pred_l2))

cross_val_score = cross_validation(X_train, y_train, k=5, alpha=0.00005, lambda_2=1000, iterations=500)
print("\nCross-validation accuracy:", cross_val_score)
