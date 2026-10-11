import torch
import numpy as np
import sys
from pathlib import Path
from embedding_transformer import Net # it says its unused but its required for torch.load
from embedding_transformer import train_loop
from embedding_transformer import user_weighted_contrasting_loss
from embedding_transformer import temp_loss

def generate():
    rng = np.random.default_rng(seed=42)
    torch.manual_seed(10)
    train = torch.randn(100, 1024)
    labels = torch.empty(100)
    user = torch.zeros(100)

    for i in range(0,100):
        labels[i] = rng.integers(0,5)
    for _ in range(3):
        one = 0
        two = 0
        while one == two:
            one = rng.integers(0, 99, endpoint=True)
            two = rng.integers(0, 99, endpoint=True)
        pos_or_neg = rng.integers(0, 1, endpoint=True)
        if pos_or_neg == 1:
            user[one] = two + 1
            user[two] = one + 1
        else:
            user[one] = 0 - two - 1
            user[two] = 0 - one - 1

    data = {
        "train": train,
        "labels": labels,
        "user": user
    }
    return data

def main():
    if len(sys.argv) < 2:
        print("invalid")
    elif sys.argv[1] == "data":
        data = generate()
        torch.save(data, str(Path.cwd()) + "/test_data/data.pt")

        return
    elif sys.argv[1] == "test":
        if len(sys.argv) < 3:
            print("invalid")
        path = sys.argv[2]
        model = torch.load(path, weights_only=False)
        data = generate()
        train = data['train']
        with torch.no_grad():
            print(model(train))
    elif sys.argv[1] == "loop":
        data = generate()
        init_data = data['train']
        model = Net(len(data['train'][0]))
        epochs = 10
        batch_size = 32
        optim = torch.optim.Adam(model.parameters(), lr = 0.1)
        train_loop(data, epochs, user_weighted_contrasting_loss, optim, batch_size, model)
        final_data = model(data['train'])
        distance = torch.linalg.vector_norm(final_data - init_data)
        distance = distance / np.sqrt(final_data.shape[-1])
        print(distance.item())


if __name__ == "__main__":
    main()