import numpy as np

# TODO kmneans cluster


class kmneans:
    # initialize K-means object
    def __init__(self, k, max_iter=100):
        self.k = k
        self.max_iter = max_iter
    # fiting the model

    def fit(self, X):
        # initialize centroids of clusters using random samples
        centroids = X[np.random.choice(X.shape[0], self.k, replace=False)]
        for _ in range(self.max_iter):
            # assign each sample to the closest centroid
            assignments = np.argmin(np.linalg.norm(
                X - centroids, axis=1), axis=0)

            # update the centroids using the mean of the samples assigned to each centroid

            # calculate the mean of the samples assigned to each centroid in new centroids to be able to check for convergence
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
