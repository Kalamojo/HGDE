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

current_dir = Path(__file__).resolve().parent
repo_root = current_dir.parent.parent
sys.path.append(str(repo_root))
from scripts.ingest_lancedb import load_lancedb_embeddings

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

def final_label(vecs, labels, user, epochs=10, model=None, optim=None):
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
        'lr': 0.005,
        'batch_size': 128,
    }
    final_vecs, model, optim = train(data, hypers, model, optim)
    final_vecs = final_vecs.numpy()
    final_labels = HDBSCAN(min_cluster_size=5).fit_predict(final_vecs)
    return final_vecs, final_labels, model, optim

def evaluate(labels, final_labels, keep_track, vecs, final_vecs, verbose=True):
    if verbose:
        print("Initial Cluster Assignments:", labels)
    print("Final Cluster Assignments:", final_labels)
    print(keep_track)
    acc = 0
    for i, j, together in keep_track:
        if verbose:
            print("==================================================")
            print("initial cluster for", i, ": ", labels[i], " now -> ", final_labels[i])
            print("initial cluster for", j, ": ", labels[j], " now -> ", final_labels[j])
        if together == 1:
            if verbose: 
                print("are they in the same cluster?", final_labels[i] == final_labels[j])
                print("are they closer?", np.linalg.norm(vecs[i]-vecs[j]) > np.linalg.norm(final_vecs[i]-final_vecs[j]))
            if final_labels[i] == final_labels[j]:
                acc += 1
        else:
            if verbose:
                print("are they in different custers?", final_labels[i] != final_labels[j])
                print("are they further apart?", np.linalg.norm(vecs[i]-vecs[j]) < np.linalg.norm(final_vecs[i]-final_vecs[j]))
            if final_labels[i] != final_labels[j]:
                acc += 1
    acc = acc / len(keep_track)
    if verbose:
        print("accuracy: ", acc)
        raw_distance = np.linalg.norm(final_vecs-vecs)
        normalized_distance = raw_distance / np.sqrt(len(vecs[0]))
        print("distance from starting", normalized_distance)
    return acc

'''
Test1 - Positive links
This test will show whether must-link vectors will be pushed closer together despite their
clusters and end up in the same cluster
'''
def test1(vecs = None, labels = None):
    if vecs is None:
        vecs, labels = make_random_vecs(100, 10)
        user = [0] * 100
    else:
        user = [0] * vecs.shape[0]
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
    final_vecs, final_labels, _, _= final_label(vecs, labels, user)
    evaluate(labels, final_labels, keep_track, vecs, final_vecs)

'''
Test2 - Negative links
This test will show whether must-link vectors will be pushed closer further apart despite their
clusters and end up in different clusters
'''
def test2(vecs = None, labels = None):
    if vecs is None:
        vecs, labels = make_random_vecs(100, 10)
        user = [0] * 100
    else:
        user = [0] * vecs.shape[0]
    keep_track = []
    for _ in range(20):
        rand1 = random.randint(0, 99)
        rand2 = random.randint(0, 99)
        while labels[rand1] != labels[rand2]:
            rand1 = random.randint(0, 99)
            rand2 = random.randint(0, 99)
        user[rand1] = 0 - (rand2+1)
        user[rand2] = 0 - (rand1+1)
        keep_track.append((rand1,rand2,0))
    final_vecs, final_labels, _, _ = final_label(vecs, labels, user)
    evaluate(labels, final_labels, keep_track, vecs, final_vecs)

'''
Test3 - Both links
Tests both links simultaneously
'''
def test3(vecs = None, labels = None):
    if vecs is None:
        vecs, labels = make_random_vecs(100, 10)
        user = [0] * 100
    else:
        user = [0] * vecs.shape[0]
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
    final_vecs, final_labels, _, _ = final_label(vecs, labels, user)
    evaluate(labels, final_labels, keep_track, vecs, final_vecs)

'''
Test4 - tests positive and negative links but trains the model one at a time to
'''
def test4(vecs = None, labels = None):
    if vecs is None:
        vecs, labels = make_random_vecs(100, 10)
        user = [0] * 100
    else:
        user = [0] * vecs.shape[0]
    init_vecs = vecs
    init_labels = labels
    keep_track = []
    model = Net(vecs.shape[1])
    lr = 0.001
    optim = torch.optim.Adam(model.parameters(), lr = lr)
    for i in range(20):
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
        vecs, labels, model, optim = final_label(vecs, labels, user, epochs=2, model=model, optim=optim)
        acc = evaluate(init_labels, labels, keep_track, init_vecs, vecs, verbose=False)
        print("loop ", i+1, " acc: ", acc)
    evaluate(init_labels, labels, keep_track, init_vecs, vecs)

    
    user = [0] * init_vecs.shape[0]
    keep_track = []
    model = Net(vecs.shape[1])
    lr = 0.001
    weight_decay = 0.00001
    optim = torch.optim.Adam(model.parameters(), lr = lr)
    for i in range(3):
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
        vecs, labels, model, optim = final_label(vecs, labels, user, epochs=10, model=model, optim=optim)
        acc = evaluate(init_labels, labels, keep_track, init_vecs, vecs, verbose=False)
        print("loop ", i, " acc: ", acc)
    evaluate(init_labels, labels, keep_track, init_vecs, vecs)


if __name__ == "__main__":
    torch.manual_seed(42)
    db_path = repo_root / "data" / "pokemon_db"
    image_embeddings = load_lancedb_embeddings(db_path=str(db_path), vector_column='image_vector')
    labels = torch.load('image_labels.pt')
    labels = labels.detach().cpu().numpy()
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
    
    # repeat the tests with the pokemon embeddings 
    print("==================================================================")
    print("============================Test 5================================")
    print("==================================================================")
    test1(vecs=image_embeddings, labels=labels)
    print("==================================================================")
    print("============================Test 6================================")
    print("==================================================================")
    test2(vecs=image_embeddings, labels=labels)
    print("==================================================================")
    print("============================Test 7================================")
    print("==================================================================")
    test3(vecs=image_embeddings, labels=labels)
    print("==================================================================")
    print("============================Test 8================================")
    print("==================================================================")
    test4(vecs=image_embeddings, labels=labels)
    