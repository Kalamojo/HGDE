import numpy as np
from numpy.typing import NDArray
from sklearn.base import ClusterMixin


class Clusterer:
    def __init__(self, method: ClusterMixin):
        self.method = method

    def run(self, embeddings: NDArray[np.float64]) -> NDArray[np.int_]:
        labels = self.method.fit_predict(embeddings)
        return np.array(labels)
