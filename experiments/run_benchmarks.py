
import sys
from pathlib import Path


current_dir = Path(__file__).resolve().parent
repo_root = current_dir.parent
sys.path.append(str(repo_root))

import src.utils.data.benchmarks as benchmarks
import src.utils.state as states
import src.utils.clusterings as clusterings
import src.utils.eval as evals
import src.utils.models as models
import src.utils.adapters as adapters
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
        python3 experiments/run_benchmarks.py --model contrastive --eval contrastive --lr 0.0001 --save_file src/utils/contrastive_run.json --rounds 20
'''

def make_string(metrics, best_metrics, best_iteration, args):
    the_string = '{"test": ' + json.dumps(vars(args)) + ',\n"final_metrics": ' + json.dumps(metrics) + ',\n"best_metrics": ' + json.dumps(best_metrics) + ',\n"best_iteration": ' + json.dumps(best_iteration) + '}\n'
    return the_string
    
# Benchmark 1 tests the given model and parameters on Red and Blue pokemon images and saves 20% for testing
def benchmark1(eval, adapter, rounds, colors, vector):
    db_path = repo_root / "data" / "pokemon_db"
    benchmark = states.ClusterState(benchmarks.PokemonBenchmark(db_path,vector,'color',0.20,categories=colors,),eval,adapter)
    metrics, best_metrics, best_iteration = benchmark.full_recluster(rounds)
    return metrics, best_metrics, best_iteration

def benchmark2(eval, adapter, rounds, label):
    db_path = repo_root / "data" / "celeba_db"
    benchmark = states.ClusterState(benchmarks.CelebaBenchmark(db_path,'vector',label,0.20,),eval,adapter)
    metrics, best_metrics, best_iteration = benchmark.full_recluster(rounds)
    return metrics, best_metrics, best_iteration

def main(eval, adapter, rounds, save_file, args):
    save_file = Path(save_file)
    if not save_file.is_absolute():
        save_file = repo_root / save_file

    the_string = '{ '
    # to add a benchmark, copy the next two lines and replace benchmark1 with your benchmark
    metrics, best_metrics, best_iteration = benchmark1(eval, adapter, rounds, ['red','blue'], 'image_vector')
    the_string += '"benchmark1": ' + make_string(metrics, best_metrics, best_iteration, args) + ',\n'
    metrics, best_metrics, best_iteration = benchmark1(eval, adapter, rounds, ['red','blue'], 'text_vector')
    the_string += '"benchmark2": ' + make_string(metrics, best_metrics, best_iteration, args) + ',\n'
    if not args.simple: # the following are not run when using the simple benchmarks for time
        metrics, best_metrics, best_iteration = benchmark1(eval, adapter, rounds, ['red','blue','green','brown'], 'image_vector')
        the_string += '"benchmark3": ' + make_string(metrics, best_metrics, best_iteration, args) + ',\n'
        metrics, best_metrics, best_iteration = benchmark1(eval, adapter, rounds, None, 'image_vector')
        the_string += '"benchmark4": ' + make_string(metrics, best_metrics, best_iteration, args) + ',\n'
        metrics, best_metrics, best_iteration = benchmark1(eval, adapter, rounds, ['red','blue','green','brown'], 'text_vector')
        the_string += '"benchmark5": ' + make_string(metrics, best_metrics, best_iteration, args) + ',\n'
        metrics, best_metrics, best_iteration = benchmark1(eval, adapter, rounds, None, 'text_vector')
        the_string += '"benchmark6": ' + make_string(metrics, best_metrics, best_iteration, args) + ',\n'
        metrics, best_metrics, best_iteration = benchmark2(eval, adapter, rounds, 'Chubby')
        the_string += '"benchmark7": ' + make_string(metrics, best_metrics, best_iteration, args) + ',\n'
        metrics, best_metrics, best_iteration = benchmark2(eval, adapter, rounds, 'Black_Hair')
        the_string += '"benchmark8": ' + make_string(metrics, best_metrics, best_iteration, args) + ',\n'
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
    parser.add_argument("--simple", action='store_true', help="runs some benchmarks to save time")
    args = parser.parse_args()

    if args.model == 'contrastive':
        adapter = adapters.JohnAdapter(clusterings.HDBSCANClusterer(), 10, 0.0001, 1024, 4096)
    elif args.model == 'triplet':
        adapter = adapters.JayAdapter(clusterings.HDBSCANClusterer(),10,0.0001,1024)
    else:
        print('unrecognized model choose between contrastive or triplet')

    if args.eval == 'contrastive':
        eval = evals.ContrastiveEval()
    elif args.eval == 'triplet':
        eval = evals.TripletEval()
    else:
        print('unrecognized eval choose between contrastive or triplet')

    
    main(eval, adapter, args.rounds, args.save_file, args)