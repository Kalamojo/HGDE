
import sys
from pathlib import Path


current_dir = Path(__file__).resolve().parent
repo_root = current_dir.parent.parent
sys.path.append(str(repo_root))

from src.utils.data.benchmarks import PokemonBenchmark
from src.utils.state import ClusterState
from src.utils.clusterings import HDBSCANClusterer
from src.utils.eval import ContrastiveEval
from src.utils.models import JohnNet
import torch

if __name__ == "__main__":
    db_path = repo_root / "data" / "pokemon_db"
    model = JohnNet(1024)
    test = ClusterState(PokemonBenchmark(db_path=db_path,vector_col="image_vector",label_col="color"),HDBSCANClusterer(),ContrastiveEval(),model,torch.optim.Adam(model.parameters(), lr = 0.001))
    test.full_recluster(10)