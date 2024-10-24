import numpy as np
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

wine_data = datasets.load_wine()
features = wine_data.data
labels = wine_data.target
train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size=0.2, random_state=42)
scaler = StandardScaler()
train_features = scaler.fit_transform(train_features)
test_features = scaler.transform(test_features)

def calculate_distance(point1, point2):
    return np.sqrt(np.sum((point1 - point2) ** 2))
class KNearestNeighbors:
    def __init__(self, num_neighbors=3):
        self.num_neighbors = num_neighbors
    
    def fit(self, train_features, train_labels):
        self.train_features = train_features
        self.train_labels = train_labels
    
    def predict(self, test_features):
        predictions = [self._predict_single(point) for point in test_features]
        return np.array(predictions)
    
    def _predict_single(self, point):
        distances = [calculate_distance(point, train_point) for train_point in self.train_features]
        neighbor_indices = np.argsort(distances)[:self.num_neighbors]
        neighbor_labels = [self.train_labels[i] for i in neighbor_indices]
        most_common = Counter(neighbor_labels).most_common(1)
        return most_common[0][0]
def calculate_accuracy(true_labels, predicted_labels):
    return np.sum(true_labels == predicted_labels) / len(true_labels)
def evaluate_knn_with_different_k_values(k_values):
    training_accuracies = []
    testing_accuracies = []
    
    for k in k_values:
        knn_model = KNearestNeighbors(num_neighbors=k)
        knn_model.fit(train_features, train_labels)

        train_predictions = knn_model.predict(train_features)
        train_accuracy = calculate_accuracy(train_labels, train_predictions)
        training_accuracies.append(train_accuracy)

        test_predictions = knn_model.predict(test_features)
        test_accuracy = calculate_accuracy(test_labels, test_predictions)
        testing_accuracies.append(test_accuracy)

        print(f"k = {k}:")
        print(f"  Training Accuracy: {train_accuracy:.2f}")
        print(f"  Testing Accuracy: {test_accuracy:.2f}\n")
    
    return training_accuracies, testing_accuracies
k_values = [1, 7, 100]
training_accuracies, testing_accuracies = evaluate_knn_with_different_k_values(k_values)

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(k_values, training_accuracies, marker='o', color='blue', label='Training Accuracy')
plt.xlabel('k')
plt.ylabel('Accuracy')
plt.title('Training Accuracy vs k')
plt.grid(True)
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(k_values, testing_accuracies, marker='o', color='red', label='Testing Accuracy')
plt.xlabel('k')
plt.ylabel('Accuracy')
plt.title('Testing Accuracy vs k')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
