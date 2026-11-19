import numpy as np
from numpy.typing import NDArray
from typing import Optional


class Predictions:
    def __init__(self):
        self.labels: Optional[NDArray[np.int_]] = None
        self.embeddings: Optional[NDArray[np.float64]] = None
    
    def update(self, labels: NDArray[np.int_], embeddings: Optional[NDArray[np.float64]] = None):
        self.labels = labels
        self.embeddings = embeddings
