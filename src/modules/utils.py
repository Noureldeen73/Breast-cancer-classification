import numpy as np


class kmneans:
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
    # fiting the model

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
            new_centroids = np.array([X[assignments == i].mean(axis=0)
                                     for i in range(self.k)])

            # check for convergence
            if np.all(centroids == new_centroids):
                break

            centroids = new_centroids

            self.centroids = centroids

        return self
    # predict the labels of the samples

    def predict(self, X):
        # assign each sample to the closest centroid
        assignments = np.argmin(np.linalg.norm(
            X - self.centroids, axis=1), axis=0)


class PCA:
    def __init__(self, n_components):
        self.n_components = n_components
        self.mean = None
        self.components = None

    def fit_transform(self, X):
        self.mean = np.mean(X, axis=0)
        X_centered = X - self.mean
        covariance_matrix = np.cov(X_centered, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)

        sorted_indices = np.argsort(eigenvalues)[::-1]
        eigenvectors = eigenvectors[:, sorted_indices]
        self.components = eigenvectors[:, :self.n_components]

        return np.dot(X_centered, self.components)
