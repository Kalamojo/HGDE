"""
description
    Unit tests for embedding model, to be expanded upon, test to see if the model truly pushes things
    together that should be together, and pushes other things apart

"""

import torch
import numpy as np
import sys
from pathlib import Path
from embedding_transformer import Net # it says its unused but its required for torch.load
from embedding_transformer import train_loop
from embedding_transformer import user_weighted_contrasting_loss
from embedding_transformer import temp_loss

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
    epochs = 10
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

if __name__ == "__main__":
    main()