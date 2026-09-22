import torch
import torch.nn as nn

from utils import augment_edges, augment_features


def train_model_graph(
    model,
    data,
    lr=0.001,
    epochs=400,
    edge_aug_mode=None,
    edge_aug_percent=0.08,
    edge_aug_is_undirected=True,
    train_feat_aug_mode=None,
    train_feat_aug_percent=0.2,
    scores=None,
):
    """Train a graph model with optional feature and edge augmentation."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    for step in range(epochs):
        model.train()

        if edge_aug_mode is None:
            edge_in = data.edge_index
        else:
            edge_kwargs = {
                "mode": edge_aug_mode,
                "percent": edge_aug_percent,
                "is_undirected": edge_aug_is_undirected,
            }
            if edge_aug_mode in {
                "bw_prob",
                "pagerank_prob",
                "degree_prob",
                "closeness_prob",
                "eigenvector_prob",
            }:
                edge_kwargs["scores"] = scores
            edge_in = augment_edges(data.edge_index, **edge_kwargs)

        if train_feat_aug_mode is None:
            train_x = data.x
        else:
            train_x = augment_features(
                data.x,
                mode=train_feat_aug_mode,
                percent=train_feat_aug_percent,
            )

        logits = model(train_x, edge_in)
        loss = loss_fn(logits[data.train_mask], data.y[data.train_mask])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 100 == 0:
            print(f"Step {step}/{epochs} loss={loss.item():.4f}", flush=True)

    return model


def train_model_base(model, data, lr=0.001, epochs=400):
    """Train a feature-only baseline model on clean node features."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    for step in range(epochs):
        model.train()
        prediction = model(data.x)

        loss = loss_fn(prediction[data.train_mask], data.y[data.train_mask])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 100 == 0:
            print(f"Step {step}, Loss: {loss.item():.4f}")

    return model
