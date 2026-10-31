from pathlib import Path
import sys
current_dir = Path(__file__).resolve().parent
repo_root = current_dir.parent.parent
sys.path.append(str(repo_root))
from tests.cluster_by_column import main

if __name__ == "__main__":
    epoch_options = [2,4,6,8]
    batch_size = [1,2,4,8,16]
    range_lr_options = range(0, 10, 1)
    lr_options = [0.00001 if lr == 0 else lr / 100000 for lr in range_lr_options]
    weight_options = [1, 2, 3, 4, 5]
    output_file = "output.csv"
    with open(output_file, "w") as file:
        pass
    with open(output_file, "a") as file:
        file.writelines("difficulty,iterations,accuracy,recall,precision,epoch_option,lr_option,batch_option,weight_option\n")
        for e in epoch_options:
            for batch in batch_size:
                for lr in lr_options:
                    for weighting in weight_options:
                        i, acc, recall, precision = main(max_corrections = 10, goal_acc = 0.95, epochs = e, lr = lr, batch_size = batch, label_column = 'color', sample_size = 100, easy = True, weighting=weighting)
                        file.writelines(f"easy,{i},{acc},{recall},{precision},{e},{lr},{batch},{weighting}\n")
                        i, acc, recall, precision = main(max_corrections = 10, goal_acc = 0.95, epochs = e, lr = lr, batch_size = batch, label_column = 'color', sample_size = 100, easy = False, weighting=weighting)
                        file.writelines(f"hard,{i},{acc},{recall},{precision},{e},{lr},{batch},{weighting}\n")