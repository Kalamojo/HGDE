import os
os.environ['LOKY_MAX_CPU_COUNT'] = '4'
import torch
import numpy as np
import sys
from pathlib import Path
import random
import numpy as np
from sklearn.cluster import HDBSCAN
from collections import Counter

current_dir = Path(__file__).resolve().parent
repo_root = current_dir.parent.parent
sys.path.append(str(repo_root))
from scripts.ingest_lancedb import load_lancedb_embeddings
from src.embedding_transformer import Net
from src.embedding_transformer import train_loop
from src.embedding_transformer import user_weighted_contrasting_loss

# computes the recall and precision of the configuation given and multiplies them to compute accuracy
def compute_accuracy(clusters, classes):
    unique_classes = set(classes)
    unique_clusters = set(clusters)
    total_precision = 1
    total_recall = 1
    class_counts = dict(Counter(classes))
    cluster_counts = dict(Counter(clusters))
    for cl in unique_classes:
        precision = 1
        recall = 1
        for cluster in unique_clusters:
            count_i_in_j = 0
            for i in range(len(classes)):
                if clusters[i] == cluster and classes[i] == cl:
                    count_i_in_j += 1
            recall *= (1 - ((count_i_in_j)/class_counts[cl]))
            precision *= (1 - ((count_i_in_j)/cluster_counts[cluster]))
        total_recall *= (1 - recall)
        total_precision *= (1 - precision)
    accuracy = total_recall * total_precision
    return accuracy, total_recall, total_precision

# pulls out the hyperparameter arguments and trains the model. returns the train final embedding
# transformation, the model, and the optimizer. Can take is model and optim so user can reuse them
def train(data, hypers, model=None, optim=None, weighting=1):
    if model == None:
        model = Net(len(data['train'][0]))
    epochs = hypers['epochs']
    lr = hypers['lr']
    batch_size = hypers['batch_size']
    if optim == None:
        optim = torch.optim.Adam(model.parameters(), lr = lr)

    train_loop(data, epochs, user_weighted_contrasting_loss, optim, batch_size, model, weighting)

    model.eval()
    with torch.no_grad():
        final_data = model(data['train'])
    return final_data, model, optim

# grabs embeddings and the given column, clusters the default embeddings and then randomly makes a correction
# and trains the model based on that
def main(max_corrections = 10, goal_acc = 0.95, epochs = 6, lr = 0.0001, batch_size = 2, label_column = 'color', sample_size = 200, easy = True, weighting=1):

    # initial setup and evaluation
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)
    db_path = repo_root / "data" / "pokemon_db"
    image_embeddings = load_lancedb_embeddings(db_path=str(db_path), vector_column='image_vector')
    true_labels = load_lancedb_embeddings(db_path=str(db_path), vector_column=label_column)
    names = load_lancedb_embeddings(db_path=str(db_path), vector_column='name')
    if easy and label_column == 'color': # easy version pulls out only red and blue pokemon
        idx = []
        for i in range(len(image_embeddings)):
            if true_labels[i] == 'red' or true_labels[i] == 'blue':
                idx.append(i)
        names = names[idx]
        image_embeddings = image_embeddings[idx]
        true_labels = true_labels[idx]
    idx = random.sample(range(len(image_embeddings)), sample_size)
    names = names[idx]
    image_embeddings = image_embeddings[idx]
    image_embeddings = np.array(image_embeddings)
    image_embeddings = image_embeddings/np.linalg.norm(image_embeddings)
    true_labels = true_labels[idx]
    pred_labels = HDBSCAN(min_cluster_size=2).fit_predict(image_embeddings)
    pred_labels = np.array(pred_labels)

    # current status of the vectors
    datapoints = []
    for i in range(sample_size):
        datapoints.append((names[i].item(), true_labels[i].item(), pred_labels[i].item()))
    print(datapoints)
    init_acc, init_recall, init_precision = compute_accuracy(pred_labels, true_labels)
    print(f"init_acc: {init_acc}, init_recall: {init_recall}, init_precision: {init_precision}")

    # set up model training
    user_specified = [0] * sample_size
    image_embeddings = torch.tensor(image_embeddings).float()
    pred_labels = torch.tensor(pred_labels)
    user_specified = torch.tensor(user_specified)
    data = {
        'train': image_embeddings,
        'labels': pred_labels,
        'user': user_specified
    }
    hypers = {
        'epochs': epochs,
        'lr': lr,
        'batch_size': batch_size
    }

    acc = 0    
    i = 0
    model = None
    optim = None
    goal = ""

    # loop of correcting and then prompting the model
    while i < max_corrections and acc < goal_acc:
        rand1 = random.randint(0, sample_size-1)
        rand2 = random.randint(0, sample_size-1)
        while (true_labels[rand1] == true_labels[rand2]) == (pred_labels[rand1] == pred_labels[rand2]):
            rand1 = random.randint(0, sample_size-1)
            rand2 = random.randint(0, sample_size-1)
        if (true_labels[rand1] == true_labels[rand2]):
            user_specified[rand1] = rand2 + 1
            user_specified[rand2] = rand1 + 1
            goal = "closer"
        else:
            user_specified[rand1] = 0 - (rand2 + 1)
            user_specified[rand2] = 0 - (rand1 + 1)
            goal = "further"
        init_distance = np.linalg.vector_norm(data['train'][rand1] - data['train'][rand2])
        data['user'] = user_specified
        final_data, model, optim = train(data, hypers, model=model, optim=optim, weighting=weighting)
        pred_labels = HDBSCAN(min_cluster_size=2).fit_predict(final_data)
        pred_labels = np.array(pred_labels)
        acc, recall, precision = compute_accuracy(pred_labels, true_labels)        
        pred_labels = torch.tensor(pred_labels)
        data['train'] = final_data
        data['labels'] = pred_labels
        if goal == 'closer':
            success = np.linalg.vector_norm(data['train'][rand1] - data['train'][rand2]) < init_distance
        else:
            success = np.linalg.vector_norm(data['train'][rand1] - data['train'][rand2]) > init_distance
        print(f"round: {i+1} - moved {rand1} and {rand2} {goal}, acc: {acc}, recall: {recall}, precision: {precision}")
        print(f"goal succeeded? {success}")
        i += 1
    iterations = i

    for i in range(sample_size):
        datapoints[i] = (datapoints[i][0], datapoints[i][1], pred_labels[i].item())
    datapoints = sorted(datapoints, key=lambda x: x[1])
    print(datapoints)

    return iterations, acc, recall, precision


if __name__ == "__main__":
    main()
