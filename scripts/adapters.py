"""
File: adapters.py
Description:
    Thin adapters that give teammates' modules the interface pipeline_controller.py expects.

        ClusteringAdapter  : wraps clustering.cluster_embeddings   -> .fit_predict(embeddings)
        ProjectionAdapter  : wraps embedding_transformer.Net       -> .transform(embeddings), .train(constraint, embeddings, labels)

    Requires the two fixes in embedding_transformer.py (remove the stray `break` in
    user_weighted_contrasting_loss, and stop train_loop shadowing `inputs`).
"""

import logging
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.preprocessing import normalize

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.append(str(repo_root))

from clustering import cluster_embeddings
from src.embedding_transformer.embedding_transformer import (
    Net,
    train_loop,
    user_weighted_contrasting_loss,
)

logger = logging.getLogger(__name__)


class ClusteringAdapter:
    """Exposes clustering.cluster_embeddings as fit_predict(embeddings) -> labels."""

    def __init__(self, n_clusters, algorithm="kmeans", random_state=123):
        self.n_clusters = n_clusters
        self.algorithm = algorithm
        self.random_state = random_state

    def fit_predict(self, embeddings):
        # clustering.py's run_* functions take different kwargs, so only pass what each accepts.
        kwargs = {}
        if self.algorithm in ("kmeans", "hierarchical", "spectral"):
            kwargs["n_clusters"] = self.n_clusters
        if self.algorithm in ("kmeans", "spectral"):
            kwargs["random_state"] = self.random_state
        labels, _model = cluster_embeddings(
            np.asarray(embeddings, dtype=np.float32), algorithm=self.algorithm, **kwargs
        )
        return labels


class ProjectionAdapter:
    """
    Wraps John's MLP so the pipeline can call transform() / train().

    - Before the first constraint, transform() is the identity (an untrained Net would just
      scramble the embeddings).
    - Constraints accumulate in John's signed, 1-indexed `user` vector:
          user[i] = +(j+1) must-link i<->j,  -(j+1) cannot-link, 0 = no constraint.
      The vector holds one partner per item, so a newer constraint on the same item overwrites
      the older one (the partner's own entry is kept).
    - Each train() call keeps training the same network/optimizer on the ORIGINAL embeddings,
      using the clustering from the current round as the base labels.
    """

    def __init__(self, dim, lr=0.005, epochs=10, batch_size=128, seed=123):
        torch.manual_seed(seed)
        self.net = Net(dim)
        self.optim = torch.optim.Adam(self.net.parameters(), lr=lr)
        self.epochs = epochs
        self.batch_size = batch_size
        self.user = None
        self.trained = False
        self.constraints = []

    @staticmethod
    def _prep(embeddings):
        # The contrastive loss (margin 2.0) assumes unit-length vectors.
        return torch.as_tensor(normalize(np.asarray(embeddings, dtype=np.float32), norm="l2"))

    def transform(self, embeddings):
        if not self.trained:
            return embeddings
        self.net.eval()  # train_loop leaves the net in train mode; dropout must be off here
        with torch.no_grad():
            return self.net(self._prep(embeddings)).numpy()

    def train(self, new_constraint, embeddings, labels):
        ctype = new_constraint["type"]
        if ctype == "MUST_LINK":
            sign = 1
        elif ctype == "CANNOT_LINK":
            sign = -1
        else:
            raise ValueError(f"Unknown constraint type: {ctype!r}")

        x = self._prep(embeddings)
        if self.user is None:
            self.user = torch.zeros(len(x), dtype=torch.long)

        i, j = int(new_constraint["item_1"]), int(new_constraint["item_2"])
        self.user[i] = sign * (j + 1)  # +1: John's format is 1-indexed because 0 means "none"
        self.user[j] = sign * (i + 1)
        self.constraints.append(new_constraint)

        data = {"train": x, "labels": torch.as_tensor(np.asarray(labels)), "user": self.user}
        train_loop(
            data, self.epochs, user_weighted_contrasting_loss, self.optim, self.batch_size, self.net
        )
        self.trained = True
        logger.debug(f"Projection model trained. Total constraints: {len(self.constraints)}")