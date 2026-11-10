from torch import optim
from pathlib import Path

import sys
current_dir = Path(__file__).resolve().parent
repo_root = current_dir.parent
sys.path.append(str(repo_root))

from utils.state import ClusterState
from utils.data.benchmarks import PokemonSampler
from utils.clusterings import HDBSCANClusterer
from utils.eval import ContrastEval, JayEval
from utils.models import JohnNet, JayNet


if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parent.parent
    vector_col = "image_vector"
    label_col = "color"
    sampler = PokemonSampler(repo_root / "data" / "pokemon_db", vector_col, label_col)
    clusterer = HDBSCANClusterer()
    eval = ContrastEval()
    epochs = 10
    model = JayNet(sampler.embeddings.shape[1])
    learning_rate = 0.001
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    rounds = 10
    cluster_state = ClusterState(sampler, clusterer, eval, epochs, model, optimizer)
    cluster_state.full_recluster(rounds)
