from torch import optim
from pathlib import Path

import sys
current_dir = Path(__file__).resolve().parent
repo_root = current_dir.parent
sys.path.append(str(repo_root))

from src.utils.state import ClusterState
from src.utils.data.samplers import PokemonSampler
from src.utils.clusterings import HDBSCANClusterer
from src.utils.eval import ContrastiveEval, TripletEval
from src.utils.models import JohnNet, JayNet


if __name__ == "__main__":
    vector_col = "image_vector"
    label_col = "color"
    batch_size = 1
    epochs = 10
    learning_rate = 0.00001
    rounds = 10

    sampler = PokemonSampler(repo_root / "data" / "pokemon_db", vector_col, label_col, test_fraction=0.2)
    clusterer = HDBSCANClusterer()
    eval = ContrastiveEval(batch_size=batch_size, epochs=epochs)
    model = JohnNet(sampler.embeddings.shape[1], dropout=0)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    
    cluster_state = ClusterState(sampler, clusterer, eval, model, optimizer)
    cluster_state.full_recluster(rounds)
