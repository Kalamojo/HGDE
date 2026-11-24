
import sys
from pathlib import Path


current_dir = Path(__file__).resolve().parent
repo_root = current_dir.parent
sys.path.append(str(repo_root))

import src.utils.data.benchmarks as benchmarks
import src.utils.state as states
import src.utils.clusterings as clusterings
import src.utils.eval as evals
import src.utils.adapters as adapters
import json
import argparse

'''
    Description:
        This file is design to run every set up benchmark with different arguments from the command line

    Editing:
        Add any benchmark using the given format and add any arguments. Just ensure that anything you add does not break
        other unrelated things.
    
    Example Usage:
        python3 experiments/run_benchmarks.py --model contrastive --eval contrastive --lr 0.0001 --save_file results/benchmark_results/contrastive_run.json --rounds 20
'''

def make_string(metrics, best_metrics, best_iteration, history, args, description='"none"'):
    the_string = '{"test": ' + json.dumps(vars(args)) + ',\n"description": ' + description + ',\n"final_metrics": ' + json.dumps(metrics) + ',\n"best_metrics": ' + json.dumps(best_metrics) + ',\n"best_iteration": ' + json.dumps(best_iteration) + ',\n"history": ' + json.dumps(history) + '}\n'
    return the_string
    
def pokemon_color_benchmark(rounds, colors, vector, args, verbose):
    if args.model == 'contrastive':
        if colors is not None:
            adapter = adapters.JohnAdapter(clusterings.KMeansClusterer(len(colors)), epochs=args.epochs, learning_rate=args.lr, dimensions=1024, hidden_size=args.hidden_size, dropout=args.dropout, verbose=args.verbose)
        else:
            adapter = adapters.JohnAdapter(clusterings.HDBSCANClusterer(), epochs=args.epochs, learning_rate=args.lr, dimensions=1024, hidden_size=args.hidden_size, dropout=args.dropout, verbose=args.verbose)
    elif args.model == 'triplet':
        if colors is not None:
            adapter = adapters.JayAdapter(clusterings.KMeansClusterer(len(colors)), epochs=args.epochs, learning_rate=args.lr, dimensions=1024, hidden_size=args.hidden_size, scaling_factor=args.scaling_factor, verbose=args.verbose)
        else:
            adapter = adapters.JayAdapter(clusterings.HDBSCANClusterer(), epochs=args.epochs, learning_rate=args.lr, dimensions=1024, hidden_size=args.hidden_size, scaling_factor=args.scaling_factor, verbose=args.verbose)
    else:
        print('unrecognized model choose between contrastive or triplet')
    if args.eval == 'contrastive':
        eval = evals.ContrastiveEval(batch_size=args.batch_size, l1_reg=args.l1_reg)
    elif args.eval == 'triplet':
        eval = evals.TripletEval(batch_size=args.batch_size, preservation_weight=args.preservation_weight)
    else:
        print('unrecognized eval choose between contrastive or triplet')
    db_path = repo_root / "data" / "pokemon_db"
    benchmark = states.ClusterState(benchmarks.PokemonBenchmark(db_path,vector,'color',0.20,categories=colors,),eval,adapter,verbose)
    metrics, best_metrics, best_iteration = benchmark.full_recluster(rounds)
    return metrics, best_metrics, best_iteration, benchmark.history   # history: every round's metrics

def pokemon_benchmark(rounds, labels, vector, args, verbose):
    if args.model == 'contrastive':
        adapter = adapters.JohnAdapter(clusterings.HDBSCANClusterer(), epochs=args.epochs, learning_rate=args.lr, dimensions=1024, hidden_size=args.hidden_size, dropout=args.dropout, verbose=args.verbose)
    elif args.model == 'triplet':
        adapter = adapters.JayAdapter(clusterings.HDBSCANClusterer(), epochs=args.epochs, learning_rate=args.lr, dimensions=1024, hidden_size=args.hidden_size, scaling_factor=args.scaling_factor, verbose=args.verbose)
    else:
        print('unrecognized model choose between contrastive or triplet')
    if args.eval == 'contrastive':
        eval = evals.ContrastiveEval(batch_size=args.batch_size, l1_reg=args.l1_reg)
    elif args.eval == 'triplet':
        eval = evals.TripletEval(batch_size=args.batch_size, preservation_weight=args.preservation_weight)
    else:
        print('unrecognized eval choose between contrastive or triplet')
    db_path = repo_root / "data" / "pokemon_db"
    benchmark = states.ClusterState(benchmarks.PokemonBenchmark(db_path,vector,labels,0.20,),eval,adapter,verbose)
    metrics, best_metrics, best_iteration = benchmark.full_recluster(rounds)
    return metrics, best_metrics, best_iteration, benchmark.history   # history: every round's metrics

def celeba_benchmark(rounds, label, args, verbose):
    if args.model == 'contrastive':
        adapter = adapters.JohnAdapter(clusterings.KMeansClusterer(2), epochs=args.epochs, learning_rate=args.lr, dimensions=1024, hidden_size=args.hidden_size, dropout=args.dropout, verbose=args.verbose)
    elif args.model == 'triplet':
            adapter = adapters.JayAdapter(clusterings.KMeansClusterer(2), epochs=args.epochs, learning_rate=args.lr, dimensions=1024, hidden_size=args.hidden_size, scaling_factor=args.scaling_factor, verbose=args.verbose)
    else:
        print('unrecognized model choose between contrastive or triplet')
    if args.eval == 'contrastive':
        eval = evals.ContrastiveEval(batch_size=args.batch_size, l1_reg=args.l1_reg)
    elif args.eval == 'triplet':
        eval = evals.TripletEval(batch_size=args.batch_size, preservation_weight=args.preservation_weight)
    else:
        print('unrecognized eval choose between contrastive or triplet')
    db_path = repo_root / "data" / "celeba_db"
    benchmark = states.ClusterState(benchmarks.CelebaBenchmark(db_path,'vector',label,0.20,),eval,adapter,verbose)
    metrics, best_metrics, best_iteration = benchmark.full_recluster(rounds)
    return metrics, best_metrics, best_iteration, benchmark.history   # history: every round's metrics

def save(save_file, the_string):
    # writes every benchmark finished so far (trailing , replaced by }), so stopping mid-run keeps them
    save_file.write_text(the_string[:-2] + '}', encoding='utf-8')

def main(rounds, save_file, args, verbpse):
    save_file = Path(save_file)
    if not save_file.is_absolute():
        save_file = repo_root / save_file
    save_file.parent.mkdir(parents=True, exist_ok=True)

    the_string = '{ '
    # to add a benchmark, copy the next two lines and replace benchmark1 with your benchmark
    metrics, best_metrics, best_iteration, history = pokemon_color_benchmark(rounds, ['red','blue'], 'image_vector', verbpse)
    the_string += '"benchmark1": ' + make_string(metrics, best_metrics, best_iteration, history, args, description='"runs image vector clustering by red and blue"') + ',\n'
    save(save_file, the_string)
    metrics, best_metrics, best_iteration, history = pokemon_color_benchmark(rounds, ['red','blue'], 'text_vector', verbpse))
    the_string += '"benchmark2": ' + make_string(metrics, best_metrics, best_iteration, history, args, description='"runs text vector clustering by red and blue"') + ',\n'
    save(save_file, the_string)
    metrics, best_metrics, best_iteration, history = pokemon_color_benchmark(rounds, ['red','blue'], ['image_vector','text_vector'], verbpse))
    the_string += '"benchmark3": ' + make_string(metrics, best_metrics, best_iteration, history, args, description='"runs image and text vector clustering by red and blue"') + ',\n'
    save(save_file, the_string)
    metrics, best_metrics, best_iteration, history = pokemon_benchmark(rounds, 'primary_type', 'image_vector', verbpse))
    the_string += '"benchmark4": ' + make_string(metrics, best_metrics, best_iteration, history, args, description='"runs image vector clustering by primary type"') + ',\n'
    save(save_file, the_string)
    metrics, best_metrics, best_iteration, history = pokemon_benchmark(rounds, 'is_baby', 'image_vector', verbpse))
    the_string += '"benchmark5": ' + make_string(metrics, best_metrics, best_iteration, history, args, description='"runs image vector clustering by baby pokemon"') + ',\n'
    save(save_file, the_string)
    if not args.simple: # the following are not run when using the simple benchmarks for time
        metrics, best_metrics, best_iteration, history = pokemon_color_benchmark(rounds, ['red','blue','green','brown'], 'image_vector', verbpse))
        the_string += '"benchmark6": ' + make_string(metrics, best_metrics, best_iteration, history, args, description='"runs image vector clustering by red, blue, green, and brown pokemon"') + ',\n'
        save(save_file, the_string)
        metrics, best_metrics, best_iteration, history = pokemon_color_benchmark(rounds, None, 'image_vector', verbpse))
        the_string += '"benchmark7": ' + make_string(metrics, best_metrics, best_iteration, history, args, description='"runs image vector clustering by all colors"') + ',\n'
        save(save_file, the_string)
        metrics, best_metrics, best_iteration, history = pokemon_color_benchmark(rounds, ['red','blue','green','brown'], 'text_vector', verbpse))
        the_string += '"benchmark8": ' + make_string(metrics, best_metrics, best_iteration, history, args, description='"runs text vector clustering by red, blue, green, and brown pokemon"') + ',\n'
        save(save_file, the_string)
        metrics, best_metrics, best_iteration, history = pokemon_color_benchmark(rounds, None, 'text_vector', verbpse))
        the_string += '"benchmark9": ' + make_string(metrics, best_metrics, best_iteration, history, args, description='"runs text vector and clusters by all colors"') + ',\n'
        save(save_file, the_string)
        metrics, best_metrics, best_iteration, history = pokemon_color_benchmark( rounds, ['red','blue','green','brown'], ['image_vector','text_vector'], verbpse))
        the_string += '"benchmark10": ' + make_string(metrics, best_metrics, best_iteration, history, args, description='"runs image and text vector clustering by red, blue, green, and brown pokemon"') + ',\n'
        save(save_file, the_string)
        metrics, best_metrics, best_iteration, history = pokemon_color_benchmark(rounds, None, ['image_vector','text_vector'], verbpse))
        the_string += '"benchmark11": ' + make_string(metrics, best_metrics, best_iteration, history, args, description='"runs image and text vector and clusters by all colors"') + ',\n'
        save(save_file, the_string)
        metrics, best_metrics, best_iteration, history = pokemon_benchmark(rounds, 'primary_type', 'text_vector', verbpse))
        the_string += '"benchmark12": ' + make_string(metrics, best_metrics, best_iteration, history, args, description='"runs text vector clustering by primary type"') + ',\n'
        save(save_file, the_string)
        metrics, best_metrics, best_iteration, history = pokemon_benchmark(rounds, 'primary_type', ['image_vector','text_vector'], verbpse))
        the_string += '"benchmark13": ' + make_string(metrics, best_metrics, best_iteration, history, args, description='"runs image and text vector clustering by primary type"') + ',\n'
        save(save_file, the_string)
        metrics, best_metrics, best_iteration, history = pokemon_benchmark(rounds, 'is_baby', 'text_vector', verbpse))
        the_string += '"benchmark14": ' + make_string(metrics, best_metrics, best_iteration, history, args, description='"runs text vector clustering by baby pokemon"') + ',\n'
        save(save_file, the_string)
        metrics, best_metrics, best_iteration, history = pokemon_benchmark(rounds, 'is_baby', ['image_vector','text_vector'], verbpse))
        the_string += '"benchmark15": ' + make_string(metrics, best_metrics, best_iteration, history, args, description='"runs image and text vector clustering by baby pokemon"') + ',\n'
        save(save_file, the_string)
        #metrics, best_metrics, best_iteration, history = celeba_benchmark(rounds, 'Chubby', verbpse))
        #the_string += '"benchmark16": ' + make_string(metrics, best_metrics, best_iteration, history, args, description='"runs celeba dataset and clusters by chubby"') + ',\n'
        #metrics, best_metrics, best_iteration, history = celeba_benchmark(rounds, 'Black_Hair', verbpse))
        #the_string += '"benchmark17": ' + make_string(metrics, best_metrics, best_iteration, history, args, description='"runs celeba dataset and clusters by black hair"') + ',\n'

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="contrastive = JohnNet, triplet = JayNet")
    parser.add_argument("--eval", required=True, help="contrastive = contrastive loss, triplet = triplet loss")
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--hidden_size", type=int, default=4096)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--scaling_factor", type=float, default=0.1)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--l1_reg", type=float, default=0.0001)
    parser.add_argument("--preservation_weight", type=float, default=0.05)
    parser.add_argument("--save_file", required=True, help="The output file to save")
    parser.add_argument("--simple", action='store_true', help="runs only some benchmarks to save time")
    parser.add_argument("--verbose", action='store_true', help="prints epoch progress")
    args = parser.parse_args()

    main(args.rounds, args.save_file, args, args.verbose)