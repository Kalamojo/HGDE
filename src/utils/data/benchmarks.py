import numpy as np
from numpy.typing import NDArray
import lancedb
import polars as pl
import random
import itertools
from pathlib import Path
from collections import defaultdict
from typing import Optional
from src.utils.modality import Modality


class Benchmark:
    def __init__(
        self, db_path: Path, table_name: str, label_col: str,
        categories: Optional[list[str]] = None, seed: int = 0
    ):
        self.db_path = db_path
        self.table_name = table_name
        self.label_col = label_col
        self.id_col = "index"
        self.seed = seed
        self.rng = random.Random(self.seed)      # all feedback sampling draws from this, so runs are reproducible
        self.categories = categories
        self.load_data()

    def load_df(self, required_columns: set[str]) -> pl.DataFrame:
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

        missing = required_columns - set(df.columns)
        if missing:
            raise ValueError(
                f"Missing {self.table_name} columns: {list(missing)}. "
                f"Available: {list(df.columns)}"
            )

        return df

    def load_members(self, labels: NDArray[np.object_], ids: NDArray[np.uint32], unique_labels: NDArray[np.object_]):
        self.true_labels = labels
        self.ids = ids
        self.unique_labels = unique_labels

    def load_data(self):
        pass;

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
            
            positives = pred_clusters[to_cluster_label]
            negatives = pred_clusters[from_cluster_label].difference({item_1})
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

class TestBenchmark(Benchmark):
    def __init__(self, db_path: Path, table_name: str, label_col: str,
                 test_fraction: float = 0.0, categories: Optional[list[str]] = None, seed: int = 0):
        self.test_fraction = test_fraction
        super().__init__(db_path, table_name, label_col, categories, seed)

    def set_test_items(self):
        """Hold out a fixed random fraction of items: they never appear in any sampled pair or
        triplet, so metrics on them measure generalization. 0.0 = no held-out items."""
        n = len(self.true_labels)
        self.test_mask = np.zeros(n, dtype=bool)
        n_test = int(round(self.test_fraction * n))
        if n_test:
            self.test_mask[np.random.default_rng(self.seed).choice(n, n_test, replace=False)] = True
        self.test_items = set(np.flatnonzero(self.test_mask).tolist())

    def load_members(self, labels: NDArray[np.object_], ids: NDArray[np.uint32], unique_labels: NDArray[np.object_]):
        super().load_members(labels, ids, unique_labels)
        self.set_test_items()

    def build_clusters(
        self, labels: NDArray[np.object_ | np.int_]
    ) -> dict[np.object_ | np.int_, set[int]]:
        clusters = defaultdict(set)

        for i in range(len(labels)):
            if i in self.test_items:
                continue;
            clusters[labels[i]].add(i)

        return clusters

class EmbeddingBenchmark(TestBenchmark):
    def __init__(self, db_path: Path, table_name: str, vector_col: str, label_col: str,
                 test_fraction: float = 0.0, categories: Optional[list[str]] = None, seed: int = 0):
        self.vector_col = vector_col
        super().__init__(db_path, table_name, label_col, test_fraction, categories,  seed)

    def load_data(self):
        required_columns = {self.vector_col, self.label_col, self.id_col}
        df = self.load_df(required_columns)

        embeddings = df[self.vector_col].to_numpy()
        labels = df[self.label_col].to_numpy()
        ids = df[self.id_col].to_numpy()

        if self.categories is not None:
            idx = []
            for i in range(len(labels)):
                if labels[i] in self.categories:
                    idx.append(i)
            embeddings = embeddings[idx]
            labels = labels[idx]
            ids = ids[idx]

        unique_labels = np.unique(labels)
        self.embeddings = embeddings
        self.load_members(labels, ids, unique_labels)

class MixedEmbeddingBenchmark(TestBenchmark):
    def __init__(self, db_path: Path, table_name: str, vector_columns: list[str], label_col: str,
                 test_fraction: float = 0.0, categories: Optional[list[str]] = None, seed: int = 0):
        self.vector_columns = vector_columns
        super().__init__(db_path, table_name, label_col, test_fraction, categories,  seed)

    def load_data(self):
        required_columns = set(self.vector_columns) | {self.label_col, self.id_col}
        df = self.load_df(required_columns)

        column_embeddings = []
        for column in self.vector_columns:
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
        labels = df[self.label_col].to_numpy()
        ids = df[self.id_col].to_numpy()

        if self.categories is not None:
            idx = []
            for i in range(len(labels)):
                if labels[i] in self.categories:
                    idx.append(i)
            embeddings = embeddings[idx]
            labels = labels[idx]
            ids = ids[idx]

        unique_labels = np.unique(labels)

        self.embeddings = embeddings
        self.load_members(labels, ids, unique_labels)

class ModalityBenchmark(TestBenchmark):
    def __init__(self, db_path: Path, table_name: str, data_col: str, modality: Modality, label_col: str,
                 test_fraction: float = 0.0, categories: Optional[list[str]] = None, seed: int = 0):
        self.data_col = data_col
        self.modality = modality
        super().__init__(db_path, table_name, label_col, test_fraction, categories,  seed)

    def load_data(self):
        required_columns = {self.data_col, self.label_col, self.id_col}
        df = self.load_df(required_columns)

        raw_data = df[self.data_col].to_numpy()
        labels = df[self.label_col].to_numpy()
        ids = df[self.id_col].to_numpy()

        if self.categories is not None:
            idx = []
            for i in range(len(labels)):
                if labels[i] in self.categories:
                    idx.append(i)
            raw_data = raw_data[idx]
            labels = labels[idx]
            ids = ids[idx]

        unique_labels = np.unique(labels)
        self.raw_data = raw_data
        self.load_members(labels, ids, unique_labels)

class PokemonBenchmark(EmbeddingBenchmark):
    def __init__(self, db_path: Path, vector_col: str, label_col: str, test_fraction: float = 0.0, categories: list[str] = None, seed: int = 0):
        table_name = "pokemon"
        super().__init__(db_path, table_name, vector_col, label_col, test_fraction, categories,  seed)

class CelebaBenchmark(EmbeddingBenchmark):
    def __init__(self, db_path: Path, vector_col: str, label_col: str, test_fraction: float = 0.0, categories: list[str] = None, seed: int = 0):
        table_name = "celeba"
        super().__init__(db_path, table_name, vector_col, label_col, test_fraction, categories,  seed)
