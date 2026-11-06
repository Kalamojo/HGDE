import os
os.environ['LOKY_MAX_CPU_COUNT'] = '4'
import torch
import numpy as np
import sys
from pathlib import Path
import random
import numpy as np
from sklearn.cluster import HDBSCAN
from sklearn.cluster import KMeans
from collections import Counter
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import LabelEncoder

current_dir = Path(__file__).resolve().parent
repo_root = current_dir.parent
sys.path.append(str(repo_root))
from scripts.ingest_lancedb import load_lancedb_embeddings
from src.embedding_transformer.embedding_transformer import Net
from src.embedding_transformer.embedding_transformer import train_loop
from src.embedding_transformer.embedding_transformer import user_weighted_contrasting_loss
from src.embedding_transformer.embedding_transformer import user_weighted_whole_cluster_loss
from src.embedding_transformer.embedding_transformer import user_weighted_goal_loss

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
    F1_score = (2*(total_recall*total_precision))/(total_recall+total_precision)
    encoder = LabelEncoder()
    encoded_labels = encoder.fit_transform(classes)
    ari_score = adjusted_rand_score(encoded_labels, clusters)
    return F1_score, total_recall, total_precision, ari_score

# pulls out the hyperparameter arguments and trains the model. returns the train final embedding
# transformation, the model, and the optimizer. Can take is model and optim so user can reuse them
def train(data, hypers, model=None, optim=None, weighting=1, loss=user_weighted_contrasting_loss, epoch_weighted=False):
    if model == None:
        model = Net(len(data['train'][0]))
    epochs = hypers['epochs']
    lr = hypers['lr']
    batch_size = hypers['batch_size']
    if optim == None:
        optim = torch.optim.Adam(model.parameters(), lr = lr)

    train_loop(data, epochs, loss, optim, batch_size, model, weighting, epoch_weighted)

    model.eval()
    with torch.no_grad():
        final_data = model(data['train'])
    return final_data, model, optim

'''
    Tests the model based on pokemon embeddings and a chosen hidden ground truth
    
    args:
        max_correction: the maximum number of moves until the model stops
        goal_acc: goal F1 score. The model will stop early if that is achieved
        epoch: number of epochs per each correction
        lr: learning rate for the model
        batch_size: batch_size
        label_column: the name of the column that gives the hidden ground truth labels
        sample_size: pulls sample_size number of random examples to run on
        easy: if true, easy mode picks a given number of labels to train on and uses k_means with a known number of clusters
        clusters: does nothing if not on easy mode, the number of clusters for easy mode to pull
        weighting: hyperparameter of user_specified weights, larger makes user_specified weighted much more
        loss: string shorthand for different loss function variations to test
        epoch_weighting: if true, the weighting given above increases by 1 for every epoch
'''
def main(max_corrections=10, goal_acc=0.95, epochs=10, lr=0.0001, batch_size=32, label_column='color', sample_size=200, easy=True, clusters=3, weighting=2, loss='whole_cluster', epoch_weighted=False):

    # initial setup and evaluation
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)
    db_path = repo_root / "data" / "pokemon_db"
    image_embeddings = load_lancedb_embeddings(db_path=str(db_path), vector_column='image_vector')
    true_labels = load_lancedb_embeddings(db_path=str(db_path), vector_column=label_column)
    names = load_lancedb_embeddings(db_path=str(db_path), vector_column='name')
    clusterer = HDBSCAN(min_cluster_size=2)
    if easy: # easy version pulls out only a given number of clusters so the optimal number of clusters is known
        label_counts = dict(Counter(true_labels))
        clusterer = KMeans(n_clusters=clusters)
        highest_keys = sorted(label_counts, key=label_counts.get, reverse=True)[:clusters]
        idx = []
        for i in range(len(image_embeddings)):
            if true_labels[i] in highest_keys:
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
    pred_labels = clusterer.fit_predict(image_embeddings)
    pred_labels = np.array(pred_labels)

    # current status of the vectors
    datapoints = []
    for i in range(sample_size):
        datapoints.append((names[i].item(), true_labels[i].item(), pred_labels[i].item()))
    print(datapoints)
    init_acc, init_recall, init_precision, init_ari_score = compute_accuracy(pred_labels, true_labels)
    print(f"init_acc: {init_acc}, init_recall: {init_recall}, init_precision: {init_precision}, init_ari_score: {init_ari_score}")

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
    if loss == 'user_weighted':
        loss_func = user_weighted_contrasting_loss
    elif loss == 'whole_cluster':
        loss_func = user_weighted_whole_cluster_loss
    elif loss == 'goal_weighted':
        loss_func = user_weighted_goal_loss

    max_acc, max_recall, max_precision, iteration = 0,0,0,0
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
        final_data, model, optim = train(data, hypers, model=model, optim=optim, weighting=weighting, loss=loss_func, epoch_weighted=epoch_weighted)
        pred_labels = clusterer.fit_predict(final_data)
        pred_labels = np.array(pred_labels)
        acc, recall, precision, ari_score = compute_accuracy(pred_labels, true_labels)        
        pred_labels = torch.tensor(pred_labels)
        data['labels'] = pred_labels
        if goal == 'closer':
            success = np.linalg.vector_norm(data['train'][rand1] - data['train'][rand2]) < init_distance
        else:
            success = np.linalg.vector_norm(data['train'][rand1] - data['train'][rand2]) > init_distance
        print(f"round: {i+1} - moved {rand1} and {rand2} {goal}, acc: {acc}, recall: {recall}, precision: {precision}, ari_score: {ari_score}")
        print(f"goal succeeded? {success}")
        i += 1
        if acc > max_acc:
            max_acc, max_recall, max_precision, iteration = acc, recall, precision, i

    for i in range(sample_size):
        datapoints[i] = (datapoints[i][0], datapoints[i][1], pred_labels[i].item())
    datapoints = sorted(datapoints, key=lambda x: x[1])
    print(datapoints)

    return iteration, max_acc, max_recall, max_precision


if __name__ == "__main__":
    main(epoch_weighted=True)
