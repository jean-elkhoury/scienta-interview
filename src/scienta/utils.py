import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors
from sknetwork.clustering import Louvain

DATA_PATH = "/Users/jelkhoury/Desktop/perso/scienta/data/pancreas.h5ad"
MLRUNS_PATH = "/Users/jelkhoury/Desktop/perso/scienta/mlruns"
RAY_RESULTS_PATH = "/Users/jelkhoury/Desktop/perso/scienta/ray_results"


def kl_divergence_gaussian(q_mean, q_log_var, p_mean, p_log_var):
    term1 = p_log_var - q_log_var
    term2 = (torch.exp(q_log_var) + (q_mean - p_mean).pow(2)) / torch.exp(p_log_var)
    kl = 0.5 * torch.sum(term1 + term2 - 1, dim=1)
    return kl


def louvain_clusters(features: np.ndarray):
    knn = NearestNeighbors(n_neighbors=10)
    knn.fit(features)
    # clust.fit(full_counts)
    graph = knn.kneighbors_graph(features)
    clust = Louvain()
    clust.fit(graph)
    return clust.labels_
