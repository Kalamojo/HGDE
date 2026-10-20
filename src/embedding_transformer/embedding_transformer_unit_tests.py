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

def train(data, hypers, model=None):
    if model == None:
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
    return final_data, model

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
Makes num random vectors of dimension d, returns the vectors and the initial labels
'''
def make_random_vecs(num, d):
    random.seed(42)
    # make num random vectors of dimension d
    vecs = []
    for _ in range(num):
        vector = []
        for _ in range(d):
            vector.append(random.randint(-10, 10))
        vecs.append(vector)
    norm = np.linalg.norm(vecs)
    vecs = vecs / norm
    label = HDBSCAN(min_cluster_size=5).fit_predict(vecs)
    return vecs, label

def final_label(vecs, labels, user, epochs=10, model=None):
    vecs = torch.tensor(vecs).float()
    labels = torch.tensor(labels)
    user = torch.tensor(user)
    data = {
        'train': vecs,
        'labels': labels,
        'user': user,
    }
    hypers = {
        'epochs': epochs,
        'lr': 0.1,
        'batch_size': 8,
        'weight_decay': 0.1
    }
    final_vecs, model = train(data, hypers, model)
    final_vecs = final_vecs.numpy()
    norm = np.linalg.norm(final_vecs)
    final_vecs = final_vecs/norm
    final_labels = HDBSCAN(min_cluster_size=5).fit_predict(final_vecs)
    return final_vecs, final_labels, model

def evaluate(labels, final_labels, keep_track, vecs, final_vecs):
    print("Initial Cluster Assignments:", labels)
    print("Final Cluster Assignments:", final_labels)
    print(keep_track)
    acc = 0
    for pair in keep_track:
        print("==================================================")
        print("initial cluster for", pair[0], ": ", labels[pair[0]], " now -> ", final_labels[pair[0]])
        print("initial cluster for", pair[1], ": ", labels[pair[1]], " now -> ", final_labels[pair[1]])
        if pair[2] == 1:
            print("are they in the same cluster?", final_labels[pair[0]] == final_labels[pair[1]])
            if final_labels[pair[0]] == final_labels[pair[1]]:
                acc += 1
        else:
            print("are they in different custers?", final_labels[pair[0]] != final_labels[pair[1]])
            if final_labels[pair[0]] != final_labels[pair[1]]:
                acc += 1
    acc = acc / len(keep_track)
    print("accuracy: ", acc)
    raw_distance = np.linalg.norm(final_vecs-vecs)
    normalized_distance = raw_distance / np.sqrt(len(vecs[0]))
    print("distance from starting", normalized_distance)

'''
Test1 - Positive links
This test will show whether must-link vectors will be pushed closer together despite their
clusters and end up in the same cluster
'''
def test1():
    vecs, labels = make_random_vecs(100, 10)
    user = [0] * 100
    keep_track = []
    for _ in range(20):
        rand1 = random.randint(0, 99)
        rand2 = random.randint(0, 99)
        while labels[rand1] == labels[rand2]:
            rand1 = random.randint(0, 99)
            rand2 = random.randint(0, 99)
        user[rand1] = rand2+1
        user[rand2] = rand1+1
        keep_track.append((rand1,rand2,1))
    final_vecs, final_labels, _ = final_label(vecs, labels, user)
    evaluate(labels, final_labels, keep_track, vecs, final_vecs)

'''
Test2 - Negative links
This test will show whether must-link vectors will be pushed closer further apart despite their
clusters and end up in different clusters
'''
def test2():
    vecs, labels = make_random_vecs(100, 10)
    user = [0] * 100
    keep_track = []
    for _ in range(20):
        rand1 = random.randint(0, 99)
        rand2 = random.randint(0, 99)
        while labels[rand1] == labels[rand2]:
            rand1 = random.randint(0, 99)
            rand2 = random.randint(0, 99)
        user[rand1] = 0 - (rand2+1)
        user[rand2] = 0 - (rand1+1)
        keep_track.append((rand1,rand2,0))
    final_vecs, final_labels, _ = final_label(vecs, labels, user)
    evaluate(labels, final_labels, keep_track, vecs, final_vecs)

'''
Test3 - Both links
Tests both links simultaneously
'''
def test3():
    vecs, labels = make_random_vecs(100, 10)
    user = [0] * 100
    keep_track = []
    for _ in range(20):
        rand1 = random.randint(0, 99)
        rand2 = random.randint(0, 99)
        if labels[rand1] == labels[rand2]:
            user[rand1] = 0 - (rand2+1)
            user[rand2] = 0 - (rand1+1)
            keep_track.append((rand1,rand2,0))
        else:
            user[rand1] = rand2+1
            user[rand2] = rand1+1
            keep_track.append((rand1,rand2,1))
    final_vecs, final_labels, _ = final_label(vecs, labels, user)
    evaluate(labels, final_labels, keep_track, vecs, final_vecs)

'''
Test4 - tests positive and negative links but trains the model one at a time to
'''
def test4():
    vecs, labels = make_random_vecs(100, 10)
    init_vecs = vecs
    init_labels = labels
    user = [0] * 100
    keep_track = []
    model = Net(10)
    for _ in range(20):
        rand1 = random.randint(0, 99)
        rand2 = random.randint(0, 99)
        if labels[rand1] == labels[rand2]:
            user[rand1] = 0 - (rand2+1)
            user[rand2] = 0 - (rand1+1)
            keep_track.append((rand1,rand2,0))
        else:
            user[rand1] = rand2+1
            user[rand2] = rand1+1
            keep_track.append((rand1,rand2,1))
        vecs, labels, model = final_label(vecs, labels, user, epochs=1, model=model)
    evaluate(init_labels, labels, keep_track, init_vecs, vecs)

    vecs, labels = make_random_vecs(100, 10)
    init_vecs = vecs
    init_labels = labels
    user = [0] * 100
    keep_track = []
    model = Net(10)
    for _ in range(3):
        rand1 = random.randint(0, 99)
        rand2 = random.randint(0, 99)
        if labels[rand1] == labels[rand2]:
            user[rand1] = 0 - (rand2+1)
            user[rand2] = 0 - (rand1+1)
            keep_track.append((rand1,rand2,0))
        else:
            user[rand1] = rand2+1
            user[rand2] = rand1+1
            keep_track.append((rand1,rand2,1))
        vecs, labels, model = final_label(vecs, labels, user, epochs=1, model=model)
    evaluate(init_labels, labels, keep_track, init_vecs, vecs)


if __name__ == "__main__":
    torch.manual_seed(42)
    print("==================================================================")
    print("============================Test 1================================")
    print("==================================================================")
    test1()
    print("==================================================================")
    print("============================Test 2================================")
    print("==================================================================")
    test2()
    print("==================================================================")
    print("============================Test 3================================")
    print("==================================================================")
    test3()
    print("==================================================================")
    print("============================Test 4================================")
    print("==================================================================")
    test4()