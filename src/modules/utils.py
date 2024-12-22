import numpy as np
import matplotlib.pyplot as plt


class kmeans:
    # initialize K-means object
    def __init__(self, k, max_iter=100):
        """
        Initialize K-means object.
        :param k: number of clusters
        :param max_iter: maximum number of iterations
        """
        self.k = k
        self.max_iter = max_iter
        self.centroids = None

    # fitting the model
    def fit(self, X):
        """
        Fit the model to the data.
        :param X: data points
        :return: self
        """
        # initialize centroids randomly from the data points
        centroids = X[np.random.choice(X.shape[0], self.k, replace=False)]
        for _ in range(self.max_iter):
            # assign each sample to the closest centroid
            distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
            assignments = np.argmin(distances, axis=1)

            # update the centroids using the mean of the samples assigned to each centroid
            new_centroids = np.array(
                [X[assignments == i].mean(axis=0) for i in range(self.k)])

            # check for convergence
            if np.allclose(centroids, new_centroids):
                break

            centroids = new_centroids

        self.centroids = centroids
        self.labels_ = assignments
        return self

    # predict the labels of the samples
    def predict(self, X):
        """
        Predict the cluster assignments for new samples
        :param X: data points
        :return: cluster assignments
        """
        distances = np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)
        assignments = np.argmin(distances, axis=1)
        return assignments

    # calculate the cost function
    def calculate_sse(self, X):
        """
        Calculate the Sum of Squared Errors (SSE) for the current clustering.
        :param X: data points
        :return: SSE value
        """
        return np.sum((X - self.centroids[self.labels_])**2)


class PCA:
    def __init__(self, n_components):
        self.n_components = n_components
        self.mean = None
        self.components = None

    def fit(self, X):
        self.mean = np.mean(X, axis=0)
        X_centered = X - self.mean
        covariance_matrix = np.cov(X_centered, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)

        sorted_indices = np.argsort(eigenvalues)[::-1]
        eigenvectors = eigenvectors[:, sorted_indices]

        self.components = eigenvectors[:, :self.n_components]

    def transform(self, X):
        X_centered = X - self.mean
        return np.dot(X_centered, self.components)


def elbow_method(X, max_k=10):
    """
    Perform the elbow method to find the optimal number of clusters.
    :param X: data points
    :param max_k: maximum number of clusters to test
    """
    sse_values = []

    # Run K-means for different values of k and calculate SSE
    for k in range(1, max_k + 1):
        km = kmeans(k, max_iter=100)
        km.fit(X)
        sse = km.calculate_sse(X)
        sse_values.append(sse)

    return sse_values


def plot_pca(data, labels=None, title="PCA Visualization"):
    plt.figure(figsize=(8, 6))
    if labels is not None:
        scatter = plt.scatter(data[:, 0], data[:, 1],
                              c=labels, cmap="viridis", s=30)
        plt.legend(handles=scatter.legend_elements()[0], title="Clusters")
    else:
        plt.scatter(data[:, 0], data[:, 1], s=30, color="blue")
    plt.title(title)
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.grid()
    plt.show()
