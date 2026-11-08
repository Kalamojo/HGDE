import numpy as np
from numpy.typing import NDArray
import lancedb
import random
import itertools
from pathlib import Path
from collections import defaultdict


class DataSampler:
    def __init__(self, db_path: Path, table_name: str, vector_col: str, label_col: str,
                 test_fraction: float = 0.0, seed: int = 0):
        self.db_path = db_path
        self.table_name = table_name
        self.vector_col = vector_col
        self.label_col = label_col
        self.id_col = "index"
        self.embeddings, self.true_labels, self.ids = self.load_data()
        self.set_test_items(test_fraction, seed)

    def set_test_items(self, test_fraction: float = 0.0, seed: int = 0):
        """Hold out a fixed random fraction of items: they never appear in any sampled pair or
        triplet, so metrics on them measure generalization. 0.0 = no held-out items."""
        n = len(self.true_labels)
        self.test_mask = np.zeros(n, dtype=bool)
        n_test = int(round(test_fraction * n))
        if n_test:
            self.test_mask[np.random.default_rng(seed).choice(n, n_test, replace=False)] = True
        self.test_items = set(np.flatnonzero(self.test_mask).tolist())

    def load_data(self) -> tuple[NDArray[np.float32], NDArray[np.object_], NDArray[np.uint32]]:
        if not self.db_path.exists():
            raise ValueError(f"LanceDB database not found at {self.db_path}")
        
        db = lancedb.connect(str(self.db_path))
        table_names = db.table_names()
        if self.table_name not in table_names:
            raise ValueError(
                f"LanceDB database at {self.db_path} "
                f"does not contain table {self.table_name}"
            )

        table = db.open_table(self.table_name)
        df = table.search().to_polars().with_row_index()

        required_columns = {self.vector_col, self.label_col, self.id_col}
        missing = required_columns - set(df.columns)
        if missing:
            raise ValueError(
                f"Missing {self.table_name} columns: {list(missing)}. "
                f"Available: {list(df.columns)}"
            )

        embeddings = df[self.vector_col].to_numpy()
        labels = df[self.label_col].to_numpy()
        ids = df[self.id_col].to_numpy()

        return embeddings, labels, ids

    def sample_pair(self, pred_labels: NDArray[np.int_]) -> tuple[int, int, bool]:
        must_links = self.fetch_must_links(pred_labels)
        must_not_links = self.fetch_must_not_links(pred_labels)
        
        all_links = must_links + must_not_links
        all_direction_flags = [(True,)] * len(must_links) + [(False,)] * len(must_not_links)
        all_pairs = [link + flag for link, flag in zip(all_links, all_direction_flags)]

        pair = random.choice(all_pairs)
        return pair

    def sample_triplets(self, pred_labels: NDArray[np.int_], count: int = 32) -> list[tuple[int, int, int]]:
        must_links = self.fetch_must_links(pred_labels)
        pred_clusters = self.build_clusters(pred_labels)

        all_moves_dict: dict[tuple[int, np.int_], tuple[int, set[int], set[int]]] = {}
        for item_1, item_2 in must_links:
            from_cluster_label = pred_labels[item_1]
            to_cluster_label = pred_labels[item_2]
            move = (item_1, to_cluster_label)
            if move in all_moves_dict:
                continue;
            
            positives = pred_clusters[to_cluster_label].difference(self.test_items)
            negatives = pred_clusters[from_cluster_label].difference({item_1}, self.test_items)
            all_moves_dict[move] = (item_1, positives, negatives)

        all_moves = [
            (item, positives, negatives)
            for item, positives, negatives in all_moves_dict.values()
            if len(negatives) > 0
        ]
        chosen_item, positives, negatives = random.choice(all_moves)

        triplet_options = list(itertools.product([chosen_item], positives, negatives))
        to_choose = min(len(triplet_options), count)
        triplets = random.sample(triplet_options, to_choose)
        return triplets

    def fetch_must_links(self, pred_labels: NDArray[np.int_]) -> list[tuple[int, int]]:
        true_clusters = self.build_clusters(self.true_labels)
        must_link = []

        for cluster in true_clusters.values():
            cluster = cluster.difference(self.test_items)
            seen = set()
            for ind1 in cluster:
                seen.add(ind1)
                for ind2 in cluster.difference(seen):
                    if pred_labels[ind1] != pred_labels[ind2]:
                        must_link.append((ind1, ind2))

        return must_link

    def fetch_must_not_links(self, pred_labels: NDArray[np.int_]) -> list[tuple[int, int]]:
        pred_clusters = self.build_clusters(pred_labels)
        must_not_link = []

        for cluster in pred_clusters.values():
            cluster = cluster.difference(self.test_items)
            seen = set()
            for ind1 in cluster:
                seen.add(ind1)
                for ind2 in cluster.difference(seen):
                    if self.true_labels[ind1] != self.true_labels[ind2]:
                        must_not_link.append((ind1, ind2))

        return must_not_link

    def build_clusters(
        self, labels: NDArray[np.object_ | np.int_]
    ) -> dict[np.object_ | np.int_, set[int]]:
        clusters = defaultdict(set)

        for i in range(len(labels)):
            clusters[labels[i]].add(i)

        return clusters

class PokemonSampler(DataSampler):
    def __init__(self, db_path: Path, vector_col: str, label_col: str, test_fraction: float = 0.0, seed: int = 0):
        table_name = "pokemon"
        super().__init__(db_path, table_name, vector_col, label_col, test_fraction, seed)
