"""
description
    Unit tests for embedding model, to be expanded upon, test to see if the model truly pushes things
    together that should be together, and pushes other things apart

"""

import os
os.environ['LOKY_MAX_CPU_COUNT'] = '4'
import torch
import numpy as np
import sys
from pathlib import Path
from embedding_transformer import Net
from embedding_transformer import train_loop
from embedding_transformer import user_weighted_contrasting_loss
from embedding_transformer import temp_loss
import random
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import numpy as np
from sklearn.cluster import HDBSCAN

def main():
    embeddings = [[10, 20, 30, 20, 30, 15], [4, 25, 15, 30, 20, 10], [-10, -12, -13, -15, -12, -11], [0, 3, 2, 5, 3, 9]]
    norm = np.linalg.norm(embeddings)
    embeddings = embeddings / norm
    embeddings = torch.tensor(embeddings).float()
    clusters = [0, 0, 1, 0]
    clusters = torch.tensor(clusters)
    user = [3, -1, 1, 0]
    user = torch.tensor(user)
    data = {
        "train": embeddings,
        "labels": clusters,
        "user": user
    }
    init_data = data['train']
    print(torch.linalg.vector_norm(init_data[0] - init_data[1]) < torch.linalg.vector_norm(init_data[0] - init_data[2]))
    print(torch.linalg.vector_norm(init_data[0] - init_data[1]).item(), " vs ", torch.linalg.vector_norm(init_data[0] - init_data[2]).item())
    model = Net(len(data['train'][0]))
    epochs = 20
    batch_size = 32
    optim = torch.optim.Adam(model.parameters(), lr = 0.01)
    train_loop(data, epochs, user_weighted_contrasting_loss, optim, batch_size, model)
    with torch.no_grad():
        final_data = model(data['train'])
    distance = torch.linalg.vector_norm(final_data - init_data)
    distance = distance / np.sqrt(final_data.shape[-1])
    print("distance from starting", distance.item())
    print(torch.linalg.vector_norm(final_data[0] - final_data[1]) > torch.linalg.vector_norm(final_data[0] - final_data[2]))
    print(torch.linalg.vector_norm(final_data[0] - final_data[1]).item(), " vs ", torch.linalg.vector_norm(final_data[0] - final_data[2]).item())
    print(torch.linalg.vector_norm(final_data[1] - final_data[3]) < torch.linalg.vector_norm(init_data[1] - init_data[3]))
    print(torch.linalg.vector_norm(final_data[1] - final_data[3]).item(), " vs ", torch.linalg.vector_norm(init_data[1] - init_data[3]).item())

    return

def train(data, hypers):
    model = Net(len(data['train'][0]))
    epochs = hypers['epochs']
    lr = hypers['lr']
    batch_size = hypers['batch_size']
    weight_decay = hypers['weight_decay']
    optim = torch.optim.AdamW(model.parameters(), lr = lr, weight_decay=weight_decay)

    train_loop(data, epochs, user_weighted_contrasting_loss, optim, batch_size, model)

    model.eval()
    with torch.no_grad():
        final_data = model(data['train'])
    return final_data

def k_clusterings(vecs):
    scores = []
    for k in range(2, 6):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(vecs)
    
        score = silhouette_score(vecs, labels)
        scores.append(score)
    k = scores.index(max(scores)) + 1
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(vecs)
    return labels

'''
Test1 - Positive links
This test will show whether must-link vectors will be pushed closer together despite their
clusters and end up in the same cluster
'''
def test1():
    random.seed(42)
    # make random vectors
    vecs = []
    for _ in range(100):
        vector = []
        for _ in range(10):
            vector.append(random.randint(-10, 10))
        vecs.append(vector)
    norm = np.linalg.norm(vecs)
    vecs = vecs / norm
    label = HDBSCAN(min_cluster_size=5).fit_predict(vecs)
    user = [0] * 100
    keep_track = []
    for _ in range(20):
        rand1 = random.randint(0, 99)
        rand2 = random.randint(0, 99)
        while label[rand1] == label[rand2]:
            rand1 = random.randint(0, 99)
            rand2 = random.randint(0, 99)
        user[rand1] = rand2
        user[rand2] = rand1
        keep_track.append((rand1,rand2))
    vecs = torch.tensor(vecs).float()
    labels = torch.tensor(label)
    user = torch.tensor(user)
    data = {
        'train': vecs,
        'labels': labels,
        'user': user,
    }
    hypers = {
        'epochs': 20,
        'lr': 0.1,
        'batch_size': 128,
        'weight_decay': 0.1
    }
    final_vecs = train(data, hypers)
    norm = np.linalg.norm(final_vecs)
    final_labels =  HDBSCAN(min_cluster_size=5).fit_predict(final_vecs)
    print("Initial Cluster Assignments:", label)
    print("Final Cluster Assignments:", final_labels)
    print(keep_track)
    acc = 0
    for pair in keep_track:
        print("==================================================")
        print("initial cluster for", pair[0], ": ", label[pair[0]])
        print("initial cluster for", pair[1], ": ", label[pair[1]])
        print("are they in the same cluster?", final_labels[pair[0]] == final_labels[pair[1]])
        print("final cluster for", pair[0], ": ", final_labels[pair[0]])
        print("final cluster for", pair[1], ": ", final_labels[pair[1]])
        if final_labels[pair[0]] == final_labels[pair[1]]:
            acc += 1
    acc = acc / len(keep_track)
    print("accuracy: ", acc)
    distance = torch.linalg.vector_norm(final_vecs - vecs)
    distance = distance / np.sqrt(final_vecs.shape[-1])
    print("distance from starting", distance.item())


'''
Test2 - Negative links
This test will show whether must-link vectors will be pushed closer further apart despite their
clusters and end up in different clusters
'''
def test2():
    random.seed(42)
    # make random vectors
    vecs = []
    for _ in range(100):
        vector = []
        for _ in range(10):
            vector.append(random.randint(-10, 10))
        vecs.append(vector)
    norm = np.linalg.norm(vecs)
    vecs = vecs / norm
    label = HDBSCAN(min_cluster_size=5).fit_predict(vecs)
    user = [0] * 100
    keep_track = []
    for _ in range(20):
        rand1 = random.randint(0, 99)
        rand2 = random.randint(0, 99)
        while label[rand1] != label[rand2]:
            rand1 = random.randint(0, 99)
            rand2 = random.randint(0, 99)
        user[rand1] = 0 - rand2
        user[rand2] = 0 - rand1
        keep_track.append((rand1,rand2))
    vecs = torch.tensor(vecs).float()
    labels = torch.tensor(label)
    user = torch.tensor(user)
    data = {
        'train': vecs,
        'labels': labels,
        'user': user,
    }
    hypers = {
        'epochs': 20,
        'lr': 0.1,
        'batch_size': 128,
        'weight_decay': 0.1
    }
    final_vecs = train(data, hypers)
    norm = np.linalg.norm(final_vecs)
    final_labels = HDBSCAN(min_cluster_size=5).fit_predict(final_vecs)
    print("Initial Cluster Assignments:", label)
    print("Final Cluster Assignments:", final_labels)
    print(keep_track)
    acc = 0
    for pair in keep_track:
        print("==================================================")
        print("initial cluster for", pair[0], ": ", label[pair[0]])
        print("initial cluster for", pair[1], ": ", label[pair[1]])
        print("are they in different custers?", final_labels[pair[0]] != final_labels[pair[1]])
        print("final cluster for", pair[0], ": ", final_labels[pair[0]])
        print("final cluster for", pair[1], ": ", final_labels[pair[1]])
        if final_labels[pair[0]] != final_labels[pair[1]]:
            acc += 1
    acc = acc / len(keep_track)
    print("accuracy: ", acc)
    distance = torch.linalg.vector_norm(final_vecs - vecs)
    distance = distance / np.sqrt(final_vecs.shape[-1])
    print("distance from starting", distance.item())

'''
Test3 - Both links
Tests both links simultaneously
'''
def test3():
    random.seed(42)
    # make random vectors
    vecs = []
    for _ in range(100):
        vector = []
        for _ in range(10):
            vector.append(random.randint(-10, 10))
        vecs.append(vector)
    norm = np.linalg.norm(vecs)
    vecs = vecs / norm
    label = HDBSCAN(min_cluster_size=5).fit_predict(vecs)
    user = [0] * 100
    keep_track = []
    for _ in range(20):
        rand1 = random.randint(0, 99)
        rand2 = random.randint(0, 99)
        if label[rand1] != label[rand2]:
            user[rand1] = rand2
            user[rand2] = rand1
            keep_track.append((rand1,rand2, 1))
        else:
            user[rand1] = rand2
            user[rand2] = rand1
            keep_track.append((rand1,rand2, 0))
    vecs = torch.tensor(vecs).float()
    labels = torch.tensor(label)
    user = torch.tensor(user)
    data = {
        'train': vecs,
        'labels': labels,
        'user': user,
    }
    hypers = {
        'epochs': 20,
        'lr': 0.1,
        'batch_size': 128,
        'weight_decay': 0.1
    }
    final_vecs = train(data, hypers)
    norm = np.linalg.norm(final_vecs)
    final_labels = HDBSCAN(min_cluster_size=5).fit_predict(final_vecs)
    print("Initial Cluster Assignments:", label)
    print("Final Cluster Assignments:", final_labels)
    print(keep_track)
    acc = 0
    for pair in keep_track:
        print("==================================================")
        print("initial cluster for", pair[0], ": ", label[pair[0]])
        print("initial cluster for", pair[1], ": ", label[pair[1]])
        if pair[2] == 1:
            print("are they in the same cluster?", final_labels[pair[0]] == final_labels[pair[1]])
            if final_labels[pair[0]] == final_labels[pair[1]]:
                acc += 1
        else:
            print("are they in different custers?", final_labels[pair[0]] != final_labels[pair[1]])
            if final_labels[pair[0]] != final_labels[pair[1]]:
                acc += 1
        print("final cluster for", pair[0], ": ", final_labels[pair[0]])
        print("final cluster for", pair[1], ": ", final_labels[pair[1]])
        
    acc = acc / len(keep_track)
    print("accuracy: ", acc)
    distance = torch.linalg.vector_norm(final_vecs - vecs)
    distance = distance / np.sqrt(final_vecs.shape[-1])
    print("distance from starting", distance.item())

if __name__ == "__main__":
    torch.manual_seed(42)
    test1()
    test2()
    test3()