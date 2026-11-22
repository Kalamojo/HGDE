import numpy as np
from numpy.typing import NDArray
from typing import Optional


class Predictions:
    def __init__(self):
        self.labels: Optional[NDArray[np.int_]] = None
        self.embeddings: Optional[NDArray[np.float64]] = None
    
    def update_labels(self, labels: NDArray[np.int_]):
        self.labels = labels
    
    def update_embeddings(self, embeddings: NDArray[np.float64]):
        self.embeddings = embeddings
