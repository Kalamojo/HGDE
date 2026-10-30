import os
os.environ['LOKY_MAX_CPU_COUNT'] = '4'
import torch
import numpy as np
import sys
from pathlib import Path
from embedding_transformer import Net
from embedding_transformer import train_loop
from embedding_transformer import user_weighted_contrasting_loss
import random
import numpy as np
from sklearn.cluster import HDBSCAN
from collections import Counter

current_dir = Path(__file__).resolve().parent
repo_root = current_dir.parent.parent
sys.path.append(str(repo_root))
from scripts.ingest_lancedb import load_lancedb_embeddings


def compute_accuracy(predictions, true_labels):
    unique_labels = set(true_labels)
    unique_predictions = set(predictions)
    total_precision = 1
    total_recall = 1
    color_counts = dict(Counter(true_labels))
    cluster_counts = dict(Counter(predictions))
    for color in unique_labels:
        precision = 1
        recall = 1
        for cluster in unique_predictions:
            count_i_in_j = 0
            for i in range(len(true_labels)):
                if predictions[i] == cluster and true_labels[i] == color:
                    count_i_in_j += 1
            recall *= (1 - (count_i_in_j)/color_counts[color])
            precision *= (1 - (count_i_in_j)/cluster_counts[cluster])
        total_recall *= (1 - recall)
        total_precision *= (1 - precision)
    accuracy = total_recall * total_precision
    return accuracy, total_recall, total_precision

def train(data, hypers, model=None, optim=None):
    if model == None:
        model = Net(len(data['train'][0]))
    epochs = hypers['epochs']
    lr = hypers['lr']
    batch_size = hypers['batch_size']
    if optim == None:
        optim = torch.optim.Adam(model.parameters(), lr = lr)

    train_loop(data, epochs, user_weighted_contrasting_loss, optim, batch_size, model)

    model.eval()
    with torch.no_grad():
        final_data = model(data['train'])
    return final_data, model, optim

def main(max_corrections = 10, goal_acc = 0.95, epochs = 6, lr = 0.0001, batch_size = 2, label_column = 'color', sample_size = 200, easy = True):

    # initial setup and evaluation
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)
    db_path = repo_root / "data" / "pokemon_db"
    image_embeddings = load_lancedb_embeddings(db_path=str(db_path), vector_column='image_vector')
    true_labels = load_lancedb_embeddings(db_path=str(db_path), vector_column=label_column)
    names = load_lancedb_embeddings(db_path=str(db_path), vector_column='name')
    if easy:
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
        final_data, model, optim = train(data, hypers, model=model, optim=optim)
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
    '''torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)
    db_path = repo_root / "data" / "pokemon_db"
    image_embeddings = load_lancedb_embeddings(db_path=str(db_path), vector_column='image_vector')
    color_labels = load_lancedb_embeddings(db_path=str(db_path), vector_column='color')
    names = load_lancedb_embeddings(db_path=str(db_path), vector_column='name')
    #idx = random.sample(range(len(image_embeddings)), 10)
    idx = []
    for i in range(len(image_embeddings)):
        if color_labels[i] == 'red' or color_labels[i] == 'blue':
            idx.append(i)
    names = names[idx]
    image_embeddings = image_embeddings[idx]
    image_embeddings = np.array(image_embeddings)
    image_embeddings = image_embeddings/np.linalg.norm(image_embeddings)
    color_labels = color_labels[idx]
    labels = HDBSCAN(min_cluster_size=2).fit_predict(image_embeddings)
    labels = np.array(labels)
    print(names)
    print(image_embeddings)
    print(color_labels)
    print(labels)

    init_acc, init_recall, init_precision = compute_accuracy(labels, color_labels)
    print(f"init_acc: {init_acc}, init_recall: {init_recall}, init_precision: {init_precision}")
    user_specified = [0] * len(image_embeddings)

    image_embeddings = torch.tensor(image_embeddings).float()
    labels = torch.tensor(labels)
    user = torch.tensor(user_specified)

    data = {
        'train': image_embeddings,
        'labels': labels,
        'user': user
    }
    hypers = {
        'epochs': 10,
        'lr': 0.001,
        'batch_size': 32
    }
    acc = 0

    i = 0
    model = None
    optim = None
    closer = "closer"
    n = len(image_embeddings) - 1
    while i < 10 and acc < 0.95:
        rand1 = random.randint(0, n)
        rand2 = random.randint(0, n)
        while (color_labels[rand1] == color_labels[rand2]) == (labels[rand1] == labels[rand2]):
            rand1 = random.randint(0, n)
            rand2 = random.randint(0, n)
        if (color_labels[rand1] == color_labels[rand2]):
            user_specified[rand1] = rand2 + 1
            user_specified[rand2] = rand1 + 1
            closer = "closer"
        else:
            user_specified[rand1] = 0 - (rand2 + 1)
            user_specified[rand2] = 0 - (rand1 + 1)
            closer = "further"
        data['user'] = user_specified
        final_data, model, optim = train(data, hypers, model=model, optim=optim)
        labels = HDBSCAN(min_cluster_size=2).fit_predict(final_data)
        labels = np.array(labels)
        acc, recall, precision = compute_accuracy(labels, color_labels)        
        labels = torch.tensor(labels)
        data['labels'] = labels
        print(f"round: {i+1} - moved {rand1} and {rand2} {closer}, acc: {acc}, recall: {recall}, precision: {precision}")
        i += 1

    print(labels)
    print(color_labels)'''
