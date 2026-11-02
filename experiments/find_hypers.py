from pathlib import Path
import sys
current_dir = Path(__file__).resolve().parent
repo_root = current_dir.parent
sys.path.append(str(repo_root))
from tests.cluster_by_column import main

if __name__ == "__main__":
    epoch_options = [2,10]
    batch_size = [32]
    lr_options = [0.00001, 0.00005, 0.0001]
    weight_options = [0,1,2,3]
    output_file = "output1.csv"
    with open(output_file, "w") as file:
        pass
    with open(output_file, "a") as file:
        file.writelines("difficulty,iterations,accuracy,recall,precision,epoch_option,lr_option,batch_option,weight_option,loss_function\n")
        for e in epoch_options:
            for batch in batch_size:
                for lr in lr_options:
                    for weighting in weight_options:
                        i, acc, recall, precision = main(max_corrections = 10, goal_acc = 0.95, epochs = e, lr = lr, batch_size = batch, label_column = 'color', sample_size = 100, easy = True, weighting=weighting, loss='user_weighted')
                        file.writelines(f"easy,{i},{acc},{recall},{precision},{e},{lr},{batch},{weighting},user_weighted\n")
                        i, acc, recall, precision = main(max_corrections = 10, goal_acc = 0.95, epochs = e, lr = lr, batch_size = batch, label_column = 'color', sample_size = 100, easy = False, weighting=weighting, loss='user_weighted')
                        file.writelines(f"hard,{i},{acc},{recall},{precision},{e},{lr},{batch},{weighting},user_weighted\n")
                        i, acc, recall, precision = main(max_corrections = 10, goal_acc = 0.95, epochs = e, lr = lr, batch_size = batch, label_column = 'color', sample_size = 100, easy = True, weighting=weighting, loss='whole_cluster')
                        file.writelines(f"easy,{i},{acc},{recall},{precision},{e},{lr},{batch},{weighting},whole_cluster\n")
                        i, acc, recall, precision = main(max_corrections = 10, goal_acc = 0.95, epochs = e, lr = lr, batch_size = batch, label_column = 'color', sample_size = 100, easy = False, weighting=weighting, loss='whole_cluster')
                        file.writelines(f"hard,{i},{acc},{recall},{precision},{e},{lr},{batch},{weighting},whole_cluster\n")