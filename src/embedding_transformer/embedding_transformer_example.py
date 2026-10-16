import torch
import numpy as np
import sys
from pathlib import Path
from embedding_transformer import Net # it might say its unused but its required for torch.load
from embedding_transformer import train_loop
from embedding_transformer import user_weighted_contrasting_loss
from embedding_transformer import temp_loss

def main():
    # Start with any embeddings that you have
    embeddings = [[10, 20, 30, 20, 30, 15], [4, 25, 15, 30, 20, 10], [-10, -12, -13, -15, -12, -11], [0, 3, 2, 5, 3, 9]]
    norm = np.linalg.norm(embeddings) # embeddings should be normalized for contrastive loss (our models might already do this)
    embeddings = embeddings / norm
    embeddings = torch.tensor(embeddings).float() # turn it to torch floats
    

    # the current cluster they are assigned to
    clusters = [0, 0, 1, 0]
    clusters = torch.tensor(clusters)

    # user specified changes
    # positive means must link (the first element must be linked with element 3, the 3rd element must be linked with element 1)
    # negative means must NOT link (the second element must not be linked with element 1)
    # 0 means the user has not made any specifications
    # IMPORTANT, because 0 is the default value, this is 1-indexed, NOT 0-indexed
    user = [3, -1, 1, 0]
    user = torch.tensor(user)

    # put the inputs into a dictionary like follows, IMPORTANT keys must be exactly this
    data = {
            "train": embeddings,
            "labels": clusters,
            "user": user
        }

    model = Net(len(data['train'][0])) # model's dimensions depend on the dimensions of the input

    # hyperparameters
    epochs = 20
    batch_size = 32
    optim = torch.optim.Adam(model.parameters(), lr = 0.01)

    # run train loop
    train_loop(data, epochs, user_weighted_contrasting_loss, optim, batch_size, model)

    # evaluate with final model
    with torch.no_grad():
        final_data = model(data['train'])

    # final_data is the initial embeddings after they were transoformed by the learned model
    print(final_data)

if __name__ == "__main__":
    main()