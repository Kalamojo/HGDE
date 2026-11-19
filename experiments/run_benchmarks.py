
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
import argparse

'''
    Description:
        This file is design to run every set up benchmark with different arguments from the command line

    Editing:
        Add any benchmark using the given format and add any arguments. Just ensure that anything you add does not break
        other unrelated things.
    
    Example Usage:
        python3 src/utils/run_benchmarks.py --model contrastive --eval contrastive --lr 0.0001 --save_file src/utils/contrastive_run.json --rounds 20
'''

def make_string(metrics, best_metrics, best_iteration, args):
    the_string = '{"test": ' + json.dumps(vars(args)) + ',\n"final_metrics": ' + json.dumps(metrics) + ',\n"best_metrics": ' + json.dumps(best_metrics) + ',\n"best_iteration": ' + json.dumps(best_iteration) + '}\n'
    return the_string
    
# Benchmark 1 tests the given model and parameters on Red and Blue pokemon images and saves 20% for testing
def benchmark1(eval, model, lr, rounds, colors, vector):
    db_path = repo_root / "data" / "pokemon_db"
    benchmark = states.ClusterState(benchmarks.PokemonBenchmark(db_path,vector,'color',0.20,categories=colors,),clusterings.HDBSCANClusterer(),eval,model,torch.optim.Adam(model.parameters(), lr=lr))
    metrics, best_metrics, best_iteration = benchmark.full_recluster(rounds)
    return metrics, best_metrics, best_iteration

def benchmark2(eval, model, lr, rounds, label):
    db_path = repo_root / "data" / "celeba_db"
    benchmark = states.ClusterState(benchmarks.CelebaBenchmark(db_path,'vector',label,0.20,),clusterings.HDBSCANClusterer(),eval,model,torch.optim.Adam(model.parameters(), lr=lr))
    metrics, best_metrics, best_iteration = benchmark.full_recluster(rounds)
    return metrics, best_metrics, best_iteration

def main(eval, model, lr, rounds, save_file, args):
    save_file = Path(save_file)
    if not save_file.is_absolute():
        save_file = repo_root / save_file

    the_string = '{ '
    # to add a benchmark, copy the next two lines and replace benchmark1 with your benchmark
    metrics, best_metrics, best_iteration = benchmark1(eval, model, lr, rounds, ['red','blue'], 'image_vector')
    the_string += '"benchmark1": ' + make_string(metrics, best_metrics, best_iteration, args) + ',\n'
    metrics, best_metrics, best_iteration = benchmark1(eval, model, lr, rounds, ['red','blue','green','brown'], 'image_vector')
    the_string += '"benchmark2": ' + make_string(metrics, best_metrics, best_iteration, args) + ',\n'
    metrics, best_metrics, best_iteration = benchmark1(eval, model, lr, rounds, None, 'image_vector')
    the_string += '"benchmark3": ' + make_string(metrics, best_metrics, best_iteration, args) + ',\n'
    metrics, best_metrics, best_iteration = benchmark1(eval, model, lr, rounds, ['red','blue'], 'text_vector')
    the_string += '"benchmark4": ' + make_string(metrics, best_metrics, best_iteration, args) + ',\n'
    metrics, best_metrics, best_iteration = benchmark1(eval, model, lr, rounds, ['red','blue','green','brown'], 'text_vector')
    the_string += '"benchmark5": ' + make_string(metrics, best_metrics, best_iteration, args) + ',\n'
    metrics, best_metrics, best_iteration = benchmark1(eval, model, lr, rounds, None, 'text_vector')
    the_string += '"benchmark6": ' + make_string(metrics, best_metrics, best_iteration, args) + ',\n'
    #metrics, best_metrics, best_iteration = benchmark2(eval, model, lr, rounds, 'Chubby')
    #the_string += '"benchmark7": ' + make_string(metrics, best_metrics, best_iteration, args) + ',\n'
    #metrics, best_metrics, best_iteration = benchmark2(eval, model, lr, rounds, 'Black_Hair')
    #the_string += '"benchmark8": ' + make_string(metrics, best_metrics, best_iteration, args) + ',\n'
    the_string = the_string[:-2] # removes trailing ,
    the_string += '}'
    save_file.parent.mkdir(parents=True, exist_ok=True)
    save_file.write_text(the_string, encoding='utf-8')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="contrastive = JohnNet, triplet = JayNet")
    parser.add_argument("--eval", required=True, help="contrastive = contrastive loss, triplet = triplet loss")
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--save_file", required=True, help="The output file to save")
    args = parser.parse_args()

    if args.model == 'contrastive':
        model = models.JohnNet(1024)
    elif args.model == 'triplet':
        model = models.JayNet(1024)
    else:
        print('unrecognized model choose between contrastive or triplet')

    if args.eval == 'contrastive':
        eval = evals.ContrastiveEval()
    elif args.eval == 'triplet':
        eval = evals.TripletEval()
    else:
        print('unrecognized eval choose between contrastive or triplet')
    
    main(eval, model, args.lr, args.rounds, args.save_file, args)