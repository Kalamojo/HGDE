import torch
import numpy as np
import sys
from pathlib import Path
from embedding_transformer import Net # it says its unused but its required for torch.load

def generate():
    rng = np.random.default_rng(seed=42)
    torch.manual_seed(10)
    train = torch.randn(100, 1024)
    labels = torch.empty(100)
    user = torch.empty(100)

    for i in range(0,100):
        labels[i] = rng.integers(0,5)
        user[i] = rng.integers(0,1, endpoint=True)
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


if __name__ == "__main__":
    main()