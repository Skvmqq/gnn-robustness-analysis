"""Dataset orchestration, result persistence, and experiment plotting."""

import csv
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch_geometric.datasets import Planetoid
from torch_geometric.datasets import Coauthor
from torch_geometric.transforms import RandomNodeSplit
from torch_geometric.datasets import Amazon

from evaluation import centrality_scores, eval_under_noise
from model import GCNModel, LogisticRegressionModel, MLPModel
from training import train_model_base, train_model_graph
from utils import (
    augment_features,
    set_seed,
)


def _plot_results(args, gcn_results, mlp_results, lr_results, output_path):
    """Save mean accuracy curves with standard-deviation error bars."""

    noise = args.noise_levels

    gcn_mean = [np.mean(x) for x in gcn_results]
    gcn_std = [np.std(x) for x in gcn_results]

    mlp_mean = [np.mean(x) for x in mlp_results]
    mlp_std = [np.std(x) for x in mlp_results]

    lr_mean = [np.mean(x) for x in lr_results]
    lr_std = [np.std(x) for x in lr_results]

    plt.figure(figsize=(8, 5))
    plt.errorbar(noise, gcn_mean, yerr=gcn_std, marker="o", capsize=5, label="GCN")
    plt.errorbar(noise, mlp_mean, yerr=mlp_std, marker="o", capsize=5, label="MLP")
    plt.errorbar(noise, lr_mean, yerr=lr_std, marker="o", capsize=5, label="LogReg")

    plt.xlabel("Noise Level")
    plt.ylabel("Test Accuracy")
    plt.title("Accuracy vs Noise Level")
    plt.legend()
    plt.grid(True)
    plt.savefig(output_path)
    plt.close()


def run_one_dataset(dataset_name, args, scenario_name="unnamed_scenario"):
    """Run one scenario for one dataset and save its result artifacts."""

    repo_root = Path(__file__).resolve().parents[1]
    data_root = repo_root / "data"

    if dataset_name in {"Cora", "CiteSeer", "Pubmed"}:
        dataset = Planetoid(root=str(data_root / dataset_name), name=dataset_name)
    elif dataset_name == "CS":
        dataset = Coauthor(root=str(data_root / "Coauthor"), name="CS")
    elif dataset_name == "Physics":
        dataset = Coauthor(root=str(data_root / "Coauthor"), name="Physics")
    elif dataset_name == "Photo":
        dataset = Amazon(root=str(data_root / "Amazon"), name="Photo")
    else:
        raise ValueError(f"Unknown dataset name: {dataset_name}")

    data = dataset[0]

    output_dir = repo_root / "results" / scenario_name / dataset_name
    output_dir.mkdir(parents=True, exist_ok=True)

    edge_aug_mode = getattr(args, "edge_aug_mode", None)
    edge_aug_percent = getattr(args, "edge_aug_percent", 0.08)
    edge_aug_is_undirected = getattr(args, "edge_aug_is_undirected", True)
    train_feat_aug_mode = getattr(args, "train_feat_aug_mode", None)
    train_feat_aug_percent = getattr(args, "train_feat_aug_percent", 0.2)

    eval_noise_mode = getattr(args, "eval_noise_mode", None) or "flip"
    use_eval_shield = getattr(args, "use_eval_shield", False)
    eval_shield_mode = getattr(args, "eval_shield_mode", None) or "fixed"
    eval_shield_threshold = getattr(args, "eval_shield_threshold", 0.15)
    eval_shield_drop_ratio = getattr(args, "eval_shield_drop_ratio", 0.1)
    eval_shield_only_when_noisy = getattr(args, "eval_shield_only_when_noisy", True)

    eval_edge_aug_mode = getattr(args, "eval_edge_aug_mode", None)
    eval_edge_aug_percent = getattr(args, "eval_edge_aug_percent", 0.1)
    eval_edge_aug_is_undirected = getattr(args, "eval_edge_aug_is_undirected", True)
    eval_apply_feature_noise = getattr(args, "eval_apply_feature_noise", True)
    eval_edge_percent_from_noise = getattr(args, "eval_edge_percent_from_noise", True)

    scores = None

    gcn_results = [[] for _ in args.noise_levels]
    mlp_results = [[] for _ in args.noise_levels]
    lr_results = [[] for _ in args.noise_levels]

    split_seed = 0
    set_seed(split_seed)
    transform = RandomNodeSplit(
        split="test_rest",
        num_train_per_class=20,
        num_val=500,
        num_test=1000,
    )
    data_run = transform(data.clone())
    scores = centrality_scores(data_run, edge_aug_mode)
    eval_scores = (
        centrality_scores(data_run, eval_edge_aug_mode) if eval_edge_aug_mode else None
    )
    for run in range(args.runs):
        print(f"\n--- Run {run + 1}/{args.runs} ---")
        set_seed(run)

        gcn = GCNModel(dataset.num_features, args, dataset.num_classes, args.dropout)
        gcn.reset_parameters()
        gcn = train_model_graph(
            gcn,
            data_run,
            args.lr,
            args.epochs,
            edge_aug_mode=edge_aug_mode,
            edge_aug_percent=edge_aug_percent,
            edge_aug_is_undirected=edge_aug_is_undirected,
            train_feat_aug_mode=train_feat_aug_mode,
            train_feat_aug_percent=train_feat_aug_percent,
            scores=scores,
        )

        mlp = MLPModel(dataset.num_features, args, dataset.num_classes, args.dropout)
        mlp.reset_parameters()
        mlp = train_model_base(mlp, data_run, args.lr, args.epochs)

        logreg = LogisticRegressionModel(max_iter=1000, random_state=run)
        logreg.fit(data_run.x[data_run.train_mask], data_run.y[data_run.train_mask])

        gcn_curve = eval_under_noise(
            gcn,
            data_run,
            args.noise_levels,
            noise_mode=eval_noise_mode,
            use_shield=use_eval_shield,
            shield_mode=eval_shield_mode,
            shield_threshold=eval_shield_threshold,
            shield_drop_ratio=eval_shield_drop_ratio,
            shield_only_when_noisy=eval_shield_only_when_noisy,
            eval_edge_aug_mode=eval_edge_aug_mode,
            eval_edge_aug_percent=eval_edge_aug_percent,
            eval_edge_aug_is_undirected=eval_edge_aug_is_undirected,
            eval_apply_feature_noise=eval_apply_feature_noise,
            eval_edge_percent_from_noise=eval_edge_percent_from_noise,
            scores=eval_scores,
        )
        mlp_curve = eval_under_noise(
            mlp,
            data_run,
            args.noise_levels,
            noise_mode=eval_noise_mode,
            eval_edge_aug_mode=eval_edge_aug_mode,
            eval_edge_aug_percent=eval_edge_aug_percent,
            eval_edge_aug_is_undirected=eval_edge_aug_is_undirected,
            eval_apply_feature_noise=eval_apply_feature_noise,
            eval_edge_percent_from_noise=eval_edge_percent_from_noise,
            scores=eval_scores,
        )

        for i, p in enumerate(args.noise_levels):
            if eval_apply_feature_noise:
                noisy_x = augment_features(data_run.x, mode=eval_noise_mode, percent=p)
            else:
                noisy_x = data_run.x
            pred_lr = logreg.predict(noisy_x[data_run.test_mask])
            pred_lr = torch.tensor(pred_lr)
            lr_acc = (pred_lr == data_run.y[data_run.test_mask]).float().mean().item()

            print(
                f"Run {run + 1}/{args.runs} | Noise={p:.1f} "
                f"GCN={gcn_curve[i]:.4f} MLP={mlp_curve[i]:.4f} LR={lr_acc:.4f}"
            )

            gcn_results[i].append(gcn_curve[i])
            mlp_results[i].append(mlp_curve[i])
            lr_results[i].append(lr_acc)

    print("\n===== FINAL (PER NOISE) =====")
    results_summary = {}
    for i, p in enumerate(args.noise_levels):
        summary = {
            "GCN": {
                "values": [float(value) for value in gcn_results[i]],
                "mean": float(np.mean(gcn_results[i])),
                "std": float(np.std(gcn_results[i])),
            },
            "MLP": {
                "values": [float(value) for value in mlp_results[i]],
                "mean": float(np.mean(mlp_results[i])),
                "std": float(np.std(mlp_results[i])),
            },
            "LogReg": {
                "values": [float(value) for value in lr_results[i]],
                "mean": float(np.mean(lr_results[i])),
                "std": float(np.std(lr_results[i])),
            },
        }
        results_summary[f"noise_{p}"] = summary
        print(
            f"Noise {p:.1f} | "
            f"GCN {np.mean(gcn_results[i]):.4f} ± {np.std(gcn_results[i]):.4f} | "
            f"MLP {np.mean(mlp_results[i]):.4f} ± {np.std(mlp_results[i]):.4f} | "
            f"LogReg {np.mean(lr_results[i]):.4f} ± {np.std(lr_results[i]):.4f}"
        )

    results_document = {
        "schema_version": 1,
        "scenario": {
            "name": scenario_name,
            "train_feature_mode": train_feat_aug_mode,
            "train_feature_percent": (
                train_feat_aug_percent if train_feat_aug_mode is not None else 0.0
            ),
            "train_edge_mode": edge_aug_mode,
            "train_edge_percent": (
                edge_aug_percent if edge_aug_mode is not None else 0.0
            ),
            "eval_feature_mode": eval_noise_mode if eval_apply_feature_noise else None,
            "eval_edge_mode": eval_edge_aug_mode,
            "eval_edge_percent": (
                eval_edge_aug_percent if eval_edge_aug_mode is not None else 0.0
            ),
        },
        "dataset": {
            "name": dataset_name,
            "num_nodes": int(data.num_nodes),
            "num_edges": int(data.edge_index.size(1)),
            "num_features": int(dataset.num_features),
            "num_classes": int(dataset.num_classes),
        },
        "training": {
            "epochs": int(args.epochs),
            "learning_rate": float(args.lr),
            "hidden_dimension": int(args.hidden),
            "dropout": float(args.dropout),
            "runs": int(args.runs),
            "seeds": list(range(args.runs)),
        },
        "evaluation": {
            "perturbation_levels": [float(level) for level in args.noise_levels],
            "metric": "test_accuracy",
            "split": {
                "type": "RandomNodeSplit",
                "train_nodes_per_class": 20,
                "validation_nodes": 500,
                "test_nodes": 1000,
                "split_seed": split_seed,
            },
        },
        "results": results_summary,
    }

    results_path = output_dir / "results.json"
    with results_path.open("w", encoding="utf-8") as f:
        json.dump(results_document, f, indent=2)

    csv_path = output_dir / "summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "scenario",
                "dataset",
                "model",
                "perturbation_level",
                "mean_accuracy",
                "std_accuracy",
            ]
        )
        for level in args.noise_levels:
            level_summary = results_summary[f"noise_{level}"]
            for model_name, model_summary in level_summary.items():
                writer.writerow(
                    [
                        scenario_name,
                        dataset_name,
                        model_name,
                        level,
                        model_summary["mean"],
                        model_summary["std"],
                    ]
                )

    plot_path = output_dir / "accuracy_plot.png"
    _plot_results(args, gcn_results, mlp_results, lr_results, plot_path)
    print(f"\nResults saved to {output_dir}")
