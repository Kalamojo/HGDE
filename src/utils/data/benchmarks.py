import numpy as np
from numpy.typing import NDArray
import lancedb
import polars as pl
import random
import itertools
from pathlib import Path
from collections import defaultdict


class Benchmark:
    def __init__(self, db_path: Path, table_name: str, vector_col: str | list[str], label_col: str,
                 test_fraction: float = 0.0, categories: list[str] = None, seed: int = 0):
        self.db_path = db_path
        self.table_name = table_name
        self.vector_col = vector_col
        self.label_col = label_col
        self.id_col = "index"
        self.seed = seed
        self.rng = random.Random(seed)      # all feedback sampling draws from this, so runs are reproducible
        self.embeddings, self.true_labels, self.ids, self.unique_labels, self.metadata = self.load_data(categories)
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

    def load_data(self, categories: list[str] = None) -> tuple[NDArray[np.float32], NDArray[np.object_], NDArray[np.uint32], NDArray[np.object_], list[dict]]:
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

        vector_columns = self.vector_col if isinstance(self.vector_col, list) else [self.vector_col]
        required_columns = set(vector_columns) | {self.label_col, self.id_col}
        missing = required_columns - set(df.columns)
        if missing:
            raise ValueError(
                f"Missing {self.table_name} columns: {list(missing)}. "
                f"Available: {list(df.columns)}"
            )

        if isinstance(self.vector_col, list):
            column_embeddings = []
            for column in self.vector_col:
                values = np.asarray(df[column].to_list())
                if values.ndim == 1:
                    values = values.reshape(-1, 1)
                column_embeddings.append(values)

            rng = np.random.default_rng(self.seed)
            selected_columns = rng.integers(0, len(column_embeddings), size=len(df))
            embeddings = np.asarray([
                column_embeddings[column_index][row_index]
                for row_index, column_index in enumerate(selected_columns)
            ], dtype=np.float32)
        else:
            embeddings = df[self.vector_col].to_numpy()
        labels = df[self.label_col].to_numpy()
        ids = df[self.id_col].to_numpy()
        # every column except the embedding vectors (e.g. dex_num, name), one dict per row
        info_columns = [c for c, t in df.schema.items()
                        if not (isinstance(t, (pl.List, pl.Array)) and t.inner.is_float())]
        metadata = df.select(info_columns).to_dicts()

        if categories is not None:
            idx = []
            for i in range(len(labels)):
                if labels[i] in categories:
                    idx.append(i)
            embeddings = embeddings[idx]
            labels = labels[idx]
            ids = ids[idx]
            metadata = [metadata[i] for i in idx]

        unique_labels = np.unique(labels)
        return embeddings, labels, ids, unique_labels, metadata

    def sample_pair(self, pred_labels: NDArray[np.int_]) -> tuple[int, int, bool] | None:
        """One (i, j, must_link) correction, or None when the clustering has nothing left to fix."""
        must_links = self.fetch_must_links(pred_labels)
        must_not_links = self.fetch_must_not_links(pred_labels)

        all_links = must_links + must_not_links
        all_direction_flags = [(True,)] * len(must_links) + [(False,)] * len(must_not_links)
        all_pairs = [link + flag for link, flag in zip(all_links, all_direction_flags)]
        if not all_pairs:
            return None

        pair = self.rng.choice(all_pairs)
        return pair

    def sample_triplets(self, pred_labels: NDArray[np.int_], count: int = 32) -> list[tuple[int, int, int]]:
        """Up to `count` (anchor, positive, negative) triplets for one item to move; [] when nothing is left to fix."""
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
        if not all_moves:
            return []
        chosen_item, positives, negatives = self.rng.choice(all_moves)

        triplet_options = list(itertools.product([chosen_item], positives, negatives))
        to_choose = min(len(triplet_options), count)
        triplets = self.rng.sample(triplet_options, to_choose)
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

class PokemonBenchmark(Benchmark):
    def __init__(self, db_path: Path, vector_col: str|list[str], label_col: str, test_fraction: float = 0.0, categories: list[str] = None, seed: int = 0):
        table_name = "pokemon"
        super().__init__(db_path, table_name, vector_col, label_col, test_fraction, categories,  seed)

class CelebaBenchmark(Benchmark):
    def __init__(self, db_path: Path, vector_col: str, label_col: str, test_fraction: float = 0.0, categories: list[str] = None, seed: int = 0):
        table_name = "celeba"
        super().__init__(db_path, table_name, vector_col, label_col, test_fraction, categories,  seed)
