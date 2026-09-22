import numpy as np
import torch

import graph_features
from utils import (
    augment_edges,
    augment_features,
    feature_similarity_rewiring,
    masked_accuracy,
)

CENTRALITY_EDGE_MODES = {
    "bw_prob",
    "pagerank_prob",
    "degree_prob",
    "closeness_prob",
    "eigenvector_prob",
}


@torch.no_grad()
def eval_under_noise(
    model,
    data,
    noise_levels,
    noise_mode="flip",
    use_shield=False,
    shield_mode="fixed",
    shield_threshold=0.15,
    shield_drop_ratio=0.1,
    shield_only_when_noisy=True,
    eval_edge_aug_mode=None,
    eval_edge_aug_percent=0.1,
    eval_edge_aug_is_undirected=True,
    eval_apply_feature_noise=True,
    eval_edge_percent_from_noise=True,
    scores=None,
):
    """Evaluate a model under feature, edge, or combined perturbations."""
    model.eval()
    accs = []

    for p in noise_levels:
        if eval_apply_feature_noise:
            noisy_x = augment_features(data.x, mode=noise_mode, percent=p)
        else:
            noisy_x = data.x

        apply_shield = use_shield and (not shield_only_when_noisy or p > 0)
        if apply_shield:
            eval_edge_index = feature_similarity_rewiring(
                noisy_x,
                data.edge_index,
                mode=shield_mode,
                threshold=shield_threshold,
                drop_ratio=shield_drop_ratio,
            )
        else:
            eval_edge_index = data.edge_index

        if eval_edge_aug_mode is not None:
            edge_kwargs = {
                "mode": eval_edge_aug_mode,
                "percent": p if eval_edge_percent_from_noise else eval_edge_aug_percent,
                "is_undirected": eval_edge_aug_is_undirected,
            }
            if eval_edge_aug_mode in CENTRALITY_EDGE_MODES:
                edge_kwargs["scores"] = scores
            eval_edge_index = augment_edges(eval_edge_index, **edge_kwargs)

        logits = model(noisy_x, eval_edge_index)
        accs.append(masked_accuracy(logits, data.y, data.test_mask))

    return np.array(accs)


def bw_centrality(data):
    """Return normalized betweenness centrality for every node."""
    gf = graph_features.GraphFeatures(data, un_directed=True)
    scores = gf.betweenness(normalized=True)
    return torch.tensor(scores, dtype=torch.float, device=data.edge_index.device)


def pagerank_centrality(data):
    """Return PageRank scores for every node."""
    gf = graph_features.GraphFeatures(data, un_directed=True)
    scores = gf.pagerank()
    return torch.tensor(scores, dtype=torch.float, device=data.edge_index.device)


def degree_centrality(data):
    """Return normalized degree centrality for every node."""
    gf = graph_features.GraphFeatures(data, un_directed=True)
    scores = gf.degree_centrality()
    return torch.tensor(scores, dtype=torch.float, device=data.edge_index.device)


def closeness_centrality(data):
    """Return closeness centrality for every node."""
    gf = graph_features.GraphFeatures(data, un_directed=True)
    scores = gf.closeness_centrality()
    return torch.tensor(scores, dtype=torch.float, device=data.edge_index.device)


def eigenvector_centrality(data):
    """Return eigenvector centrality, with a zero fallback on failure."""
    gf = graph_features.GraphFeatures(data, un_directed=True)
    try:
        scores = gf.eigenvector_centrality()
    except Exception:
        scores = [0.0] * data.num_nodes
    return torch.tensor(scores, dtype=torch.float, device=data.edge_index.device)


def centrality_scores(data, edge_aug_mode):
    """Compute node scores for the selected topology-aware edge mode."""
    score_functions = {
        "bw_prob": bw_centrality,
        "pagerank_prob": pagerank_centrality,
        "degree_prob": degree_centrality,
        "closeness_prob": closeness_centrality,
        "eigenvector_prob": eigenvector_centrality,
    }
    score_function = score_functions.get(edge_aug_mode)
    return score_function(data) if score_function else None
