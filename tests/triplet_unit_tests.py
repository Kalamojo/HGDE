# Unit tests of triplet learning
import matplotlib.pyplot as plt

# Test performance on KMeans, DBScan, Hierarchial, and Spectral
def testing_by_clustering_method():
    # Graph performance of each clustering method
    clustering_methods = ['KMeans', 'DBScan', 'Hierarchical', 'Spectral']
    for method in clustering_methods:
        ...
    return None

# Test performance on text vs images vs mixed
def testing_by_modality():
    modalities = ['text', 'image', 'mixed']
    return None

# Test performance by number of feedback rounds
def test_by_num_feedback_rounds():
    num_feedback_rounds = [3, 5, 10, 20, 50]
    return None

# Test performance by batch size
def test_by_batch_size():
    batch_sizes = [16, 32, 64, 128, 256]
    return None

# Test performance by different learning rates
def test_by_learning_rate():
    learning_rates = [0.0001, 0.001, 0.01, 0.05, 0.1, 0.3]
    return None

# Test performance by scaling factor for optimizer
def test_by_scaling_factor():
    scaling_factors = [0.1, 0.5, 1, 2, 5]
    return None

# Test performance on preservation loss weight
def test_by_preservation_loss_weight():
    preservation_loss_weights = [0.1, 0.5, 1, 2, 5]
    return None