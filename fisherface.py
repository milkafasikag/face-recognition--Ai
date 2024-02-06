import numpy as np
from collections import Counter


def knn_classifier(projections, labels, x, k):
    distances = [np.linalg.norm(x - pi) for pi in projections]
    indices = np.argsort(distances)[:k]

    k_nearest_labels = [labels[i] for i in indices]
    majority_vote = Counter(k_nearest_labels).most_common(1)

    label = majority_vote[0][0]

    return label


def apply_pca(X, num_components=15):
    n, _ = X.shape

    mean = np.mean(X, axis=0)

    centered_data = X - mean

    L = np.dot(centered_data, centered_data.T)

    eigenvalues, eigenvectors = np.linalg.eig(L)

    eigenvectors = np.dot(centered_data.T, eigenvectors)

    for i in range(n):
        eigenvectors[:, i] = eigenvectors[:, i] / np.linalg.norm(eigenvectors[:, i])

    idx = np.argsort(-eigenvalues)
    eigenvalues, eigenvectors = eigenvalues[idx], eigenvectors[:, idx]

    return eigenvalues[:num_components], eigenvectors[:, :num_components], mean


def apply_lda(X, y):
    classes = np.unique(y)

    n, d = X.shape

    total_mean = X.mean(axis=0)

    # with in class scatter matrix
    sw = np.zeros((d, d))

    # between class scatter matrix
    sb = np.zeros((d, d))

    for label in classes:
        group = X[np.where(y == label)[0], :]

        group_mean = group.mean(axis=0)

        sw = sw + np.dot((group - group_mean).T, group - group_mean)

        sb = sb + n * np.dot((group_mean - total_mean).T, (group_mean - total_mean))

    eigenvalues, eigenvectors = np.linalg.eig(np.linalg.inv(sw) * sb)

    idx = np.argsort(-eigenvalues.real)
    eigenvalues, eigenvectors = eigenvalues[idx].real, eigenvectors[:, idx].real

    return eigenvalues, eigenvectors


def project(W, X, mean):
    return np.dot(X - mean, W)


def reconstruct(W, Y):
    return np.dot(Y, W.T)


def compute_threshold(projections):
    n = len(projections)

    max_dist = float("-inf")

    for i in range(n):
        for j in range(i + 1, n):
            dist = np.linalg.norm(projections[i] - projections[j])
            max_dist = max(max_dist, dist)

    return max_dist


def closest_face(projections, projection_labels, p):
    distances = [np.linalg.norm(p - pi) for pi in projections]

    sorted_distances_idx = np.argsort(distances)

    return (
        projection_labels[sorted_distances_idx[0]],
        distances[sorted_distances_idx[0]],
    )


class FisherFaces:
    def __init__(self, k=3, num_components=15):
        self.k = k
        self.num_components = num_components

    def fit(self, x_train, y_train):
        self.projection_labels = y_train

        eigenvalues_pca, eigenvectors_pca, self.mean_face = apply_pca(
            x_train, num_components=self.num_components
        )

        self.eigenvalues_lda, eigenvectors_lda = apply_lda(
            project(eigenvectors_pca, x_train, self.mean_face), y_train
        )

        self.eigenvectors = np.dot(eigenvectors_pca, eigenvectors_lda)

        projections = []

        for image in x_train:
            projections.append(
                project(self.eigenvectors, image.reshape(1, -1), self.mean_face)
            )

        self.projections = np.array(projections)

        return self

    def transform(self, X):
        y_pred = []

        for image in X:
            projected_image = project(
                self.eigenvectors, image.reshape(1, -1), self.mean_face
            )

            prediction = knn_classifier(
                self.projections, self.projection_labels, projected_image, self.k
            )
            y_pred.append(prediction)

        return np.array(y_pred)
