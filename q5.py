import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler

X, y = make_blobs(n_samples=100, centers=2, random_state=42, cluster_std=1.5)
y = np.where(y == 0, -1, 1)
scaler = StandardScaler()
X = scaler.fit_transform(X)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='Spectral')
plt.title("Final Dataset")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()

def perceptronfunc(X, y, w, b):
    loss = np.maximum(0, -y * (np.dot(X, w) + b))
    return np.sum(loss)
def hingelossfunc(X, y, w, b):
    loss = np.maximum(0, 1 - y * (np.dot(X, w) + b))
    return np.sum(loss)
def subgradientdescent(X, y, loss_fn, w_init, b_init, learning_rate=0.01, num_epochs=100):
    w = w_init.copy()
    b = b_init
    losses = []

    for epoch in range(num_epochs):
        total_loss = 0
        for i in range(X.shape[0]):
            prediction = np.dot(X[i], w) + b
            loss = loss_fn(X[i:i+1], y[i:i+1], w, b)
            total_loss += loss
            if loss_fn == perceptronfunc:
                if loss > 0:
                    w += learning_rate * (y[i] * X[i])
                    b += learning_rate * y[i]
            elif loss_fn == hingelossfunc:
                if 1 - y[i] * prediction > 0:
                    w += learning_rate * (y[i] * X[i])
                    b += learning_rate * y[i]

        losses.append(total_loss)


        if epoch % 10 == 0:
            print(f'Epoch {epoch}, Total Loss: {total_loss:.2f}, Weights: {w}, Bias: {b:.2f}')

    return w, b, losses


np.random.seed(42)
w_init = np.random.rand(X.shape[1])
b_init = np.random.rand(1)[0]
w_perceptron, b_perceptron, perceptron_losses = subgradientdescent(X, y, perceptronfunc, w_init, b_init, learning_rate=0.01, num_epochs=100)
w_hinge, b_hinge, hinge_losses = subgradientdescent(X, y, hingelossfunc, w_init, b_init, learning_rate=0.01, num_epochs=100)
train_predictions_perceptron = np.sign(np.dot(X, w_perceptron) + b_perceptron)
train_predictions_hinge = np.sign(np.dot(X, w_hinge) + b_hinge)

print("Perceptron Model Parameters:")
print(f"Weights: {w_perceptron}")
print(f"Bias: {b_perceptron:.2f}")
print(f"Final Perceptron Loss: {perceptron_losses[-1]:.2f}")
print("\nHinge Loss SVM Model Parameters:")
print(f"Weights: {w_hinge}")
print(f"Bias: {b_hinge:.2f}")
print(f"Final Hinge Loss: {hinge_losses[-1]:.2f}")

plt.figure(figsize=(12, 5))
plt.plot(perceptron_losses, label='Perceptron Loss', color='blue')
plt.plot(hinge_losses, label='Hinge Loss', color='red')
plt.title("Loss Curves")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()
def plotdecisionboundary(X, y, w, b, title="Decision Boundary"):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                         np.arange(y_min, y_max, 0.02))
    Z = np.dot(np.c_[xx.ravel(), yy.ravel()], w) + b
    Z = np.where(Z >= 0, 1, -1)
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, alpha=0.8)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k')
    plt.title("Question five")
    plt.show()
plotdecisionboundary(X, y, w_perceptron, b_perceptron, title="Perceptron Output")
plotdecisionboundary(X, y, w_hinge, b_hinge, title="Hinge-Loss SVM Output")