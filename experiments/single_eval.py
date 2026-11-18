from pathlib import Path

import sys
current_dir = Path(__file__).resolve().parent
repo_root = current_dir.parent
sys.path.append(str(repo_root))

from src.utils.state import ClusterState
from src.utils.data.benchmarks import PokemonBenchmark
from src.utils.clusterings import HDBSCANClusterer, KMeansClusterer
from src.utils.eval import ContrastiveEval, TripletEval
from src.utils.adapters import JohnAdapter, JayAdapter


if __name__ == "__main__":
    vector_col = "text_vector"
    label_col = "color"
    batch_size = 16
    epochs = 20
    learning_rate = 0.001
    rounds = 10

    benchmark = PokemonBenchmark(repo_root / "data" / "pokemon_db", vector_col, label_col, test_fraction=0.2, seed=42)
    clusterer = KMeansClusterer(benchmark.unique_labels.shape[0])
    eval = TripletEval(batch_size)
    adapter = JayAdapter(epochs, learning_rate, benchmark.embeddings.shape[1])
    
    
    cluster_state = ClusterState(benchmark, clusterer, eval, adapter)
    cluster_state.full_recluster(rounds)
