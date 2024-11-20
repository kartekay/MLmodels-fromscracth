import numpy as np
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

digits = load_digits()
data = digits.data

def k_means_pp_with_loss(data, k, max_iters=100):
    np.random.seed(42)
    n_samples, n_features = data.shape
    centroids = [data[np.random.randint(n_samples)]]
    for _ in range(1, k):
        distances = np.min(np.linalg.norm(data[:, None] - centroids, axis=2), axis=1)
        probabilities = distances / np.sum(distances)
        next_centroid = data[np.random.choice(range(n_samples), p=probabilities)]
        centroids.append(next_centroid)
    centroids = np.array(centroids)
    losses = []

    for _ in range(max_iters):
        labels = np.argmin(np.linalg.norm(data[:, None] - centroids, axis=2), axis=1)
        new_centroids = np.array([data[labels == i].mean(axis=0) for i in range(k)])
        loss = np.sum([np.linalg.norm(data[labels == i] - centroid) for i, centroid in enumerate(new_centroids)])
        losses.append(loss)
        if np.allclose(centroids, new_centroids):
            break
        centroids = new_centroids

    return labels, centroids, losses

def k_medoids_plus_with_loss(data, k, max_iters=100):
    np.random.seed(42)
    n_samples = data.shape[0]
    medoids = [np.random.randint(n_samples)]
    for _ in range(1, k):
        distances = np.min(np.linalg.norm(data[:, None] - data[medoids], axis=2), axis=1)
        next_medoid = np.argmax(distances)
        medoids.append(next_medoid)
    medoids = np.array(medoids)
    labels = np.zeros(n_samples)
    losses = []

    for _ in range(max_iters):
        distances = np.linalg.norm(data[:, None] - data[medoids], axis=2)
        labels = np.argmin(distances, axis=1)
        loss = np.sum(np.min(distances, axis=1))
        losses.append(loss)

        new_medoids = []
        for i in range(k):
            cluster_points = data[labels == i]
            if len(cluster_points) > 0:
                distance_sums = np.sum(np.linalg.norm(cluster_points[:, None] - cluster_points, axis=2), axis=0)
                new_medoid = np.argmin(distance_sums)
                new_medoids.append(new_medoid)
            else:
                new_medoids.append(medoids[i])
        new_medoids = np.array(new_medoids)

        if np.array_equal(medoids, new_medoids):
            break
        medoids = new_medoids

    return labels, medoids, losses

k = 10

kmeans_labels, kmeans_centroids, kmeans_losses = k_means_pp_with_loss(data, k)
kmedoids_labels, kmedoids_medoids, kmedoids_losses = k_medoids_plus_with_loss(data, k)

pca = PCA(n_components=2)
reduced_data = pca.fit_transform(data)

plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=kmeans_labels, cmap='tab10', s=10)
plt.title("Enhanced K-Means Clustering")

plt.subplot(1, 2, 2)
plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=kmedoids_labels, cmap='tab10', s=10)
plt.title("Enhanced K-Medoids Clustering")

plt.show()

plt.figure(figsize=(10, 5))
plt.plot(range(1, len(kmeans_losses) + 1), kmeans_losses, label="K-Means Loss", marker="o")
plt.plot(range(1, len(kmedoids_losses) + 1), kmedoids_losses, label="K-Medoids Loss", marker="o")
plt.xlabel("Iteration")
plt.ylabel("Clustering Loss")
plt.title("Loss Reduction in K-Means and K-Medoids")
plt.legend()
plt.show()

print(f"Final K-Means Loss: {kmeans_losses[-1]}")
print(f"Final K-Medoids Loss: {kmedoids_losses[-1]}")
