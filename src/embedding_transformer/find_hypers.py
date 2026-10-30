from cluster_by_color import main

if __name__ == "__main__":
    epoch_options = [4]
    range_lr_options = range(0, 10, 1)
    lr_options = [0.00001 if lr == 0 else lr / 100000 for lr in range_lr_options]
    batch_size = [16]
    weight_options = [1, 2, 3, 4, 5]
    output_file = "output2.txt"
    with open(output_file, "w") as file:
        pass
    with open(output_file, "a") as file:
        for e in epoch_options:
            for lr in lr_options:
                for batch in batch_size:
                    for weighting in weight_options:
                        i, acc, recall, precision = main(max_corrections = 10, goal_acc = 0.95, epochs = e, lr = lr, batch_size = batch, label_column = 'color', sample_size = 100, easy = True, weighting=weighting)
                        file.writelines(f"easy -> iteration: {i} acc: {acc} recall: {recall} precision: {precision} epoch_option: {e} lr_option: {lr} batch_option: {batch} weight_option: {weighting}\n")
                        i, acc, recall, precision = main(max_corrections = 10, goal_acc = 0.95, epochs = e, lr = lr, batch_size = batch, label_column = 'color', sample_size = 100, easy = False, weighting=weighting)
                        file.writelines(f"hard -> iteration: {i} acc: {acc} recall: {recall} precision: {precision} epoch_option: {e} lr_option: {lr} batch_option: {batch} weight_option: {weighting}\n")