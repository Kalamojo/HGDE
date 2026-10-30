from cluster_by_color import main

if __name__ == "__main__":
    epoch_options = [2, 4, 6, 8, 10]
    range_lr_options = range(0, 100, 5)
    lr_options = [0.00001 if lr == 0 else lr / 100000 for lr in range_lr_options]
    batch_size = [1, 2, 4, 8, 16, 32]
    output_file = "output.txt"
    with open(output_file, "w") as file:
        pass
    with open(output_file, "a") as file:
        for e in epoch_options:
            for lr in lr_options:
                for batch in batch_size:
                    i, acc, recall, precision = main(max_corrections = 10, goal_acc = 0.95, epochs = e, lr = lr, batch_size = batch, label_column = 'color', sample_size = 100, easy = True)
                    file.writelines(f"easy -> iteration: {i} acc: {acc} recall: {recall} precision: {precision} epoch_option: {e} lr_option: {lr} batch_option: {batch} \n")
                    i, acc, recall, precision = main(max_corrections = 10, goal_acc = 0.95, epochs = e, lr = lr, batch_size = batch, label_column = 'color', sample_size = 100, easy = False)
                    file.writelines(f"hard -> iteration: {i} acc: {acc} recall: {recall} precision: {precision} epoch_option: {e} lr_option: {lr} batch_option: {batch} \n")