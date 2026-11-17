
import sys
from pathlib import Path


current_dir = Path(__file__).resolve().parent
repo_root = current_dir.parent.parent
sys.path.append(str(repo_root))

import src.utils.data.benchmarks as benchmarks
import src.utils.state as states
import src.utils.clusterings as clusterings
import src.utils.eval as evals
import src.utils.models as models
import torch
import json

def make_string(metrics, best_metrics):
    the_string = '{"final_metrics": ' + json.dumps(metrics) + ',' + '"best_metrics": ' + json.dumps(best_metrics) + '}\n'
    return the_string
    
# Benchmark 1 tests the given model and parameters on Red and Blue pokemon images and saves 20% for testing
def benchmark1(eval, model, lr):
    db_path = repo_root / "data" / "pokemon_db"
    benchmark = states.ClusterState(benchmarks.PokemonRedBlueBenchmark(db_path,'image_vector',0.20,),clusterings.HDBSCANClusterer(),eval,model,torch.optim.Adam(model.parameters(), lr=lr))
    metrics, best_metrics = benchmark.full_recluster(10)
    return metrics, best_metrics

def main(eval, model, lr, save_path):
    save_path = Path(save_path)
    if not save_path.is_absolute():
        save_path = repo_root / 'src' / 'utils' / save_path

    metrics, best_metrics = benchmark1(eval, model, lr)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    save_path.write_text(make_string(metrics, best_metrics), encoding='utf-8')

if __name__ == "__main__":
    eval = evals.ContrastiveEval()
    model = models.JohnNet(1024, 4096)
    lr = 0.001
    save_path = 'stats.json'
    main(eval, model, lr, save_path)