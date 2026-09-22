"""Command-line entry point for the complete experiment suite."""

import torch

from experiment_runner import run_one_dataset
from scenarios import SCENARIOS, describe_scenario


class BaseArgs:
    datasets = ["Cora", "CiteSeer", "Pubmed", "CS", "Physics", "Photo"]
    hidden = 64
    dropout = 0.5
    lr = 0.001
    epochs = 400
    runs = 10
    noise_levels = [0.0, 0.2, 0.4, 0.6, 0.9]
    eval_noise_mode = "noise"
    use_eval_shield = False
    eval_shield_mode = "fixed"
    eval_shield_threshold = 0.15
    eval_shield_drop_ratio = 0.10
    eval_shield_only_when_noisy = True


def run_suite():
    """Run every configured scenario across every configured dataset."""

    for scenario in SCENARIOS:
        print(f"\n{'#' * 30}")
        print(f"STARTING SCENARIO: {scenario['name']}")
        print(f"PURPOSE: {describe_scenario(scenario)}")
        print(f"{'#' * 30}\n")

        args = BaseArgs()
        args.train_feat_aug_mode = scenario["feat_mode"]
        args.train_feat_aug_percent = 0.1
        args.edge_aug_mode = scenario["edge_mode"]
        args.edge_aug_percent = 0.1
        args.edge_aug_is_undirected = True
        args.eval_edge_aug_mode = scenario["eval_edge_mode"]
        args.eval_edge_aug_percent = 0.1
        args.eval_edge_aug_is_undirected = True
        args.eval_apply_feature_noise = scenario["eval_feature_noise"]
        args.eval_edge_percent_from_noise = True

        for dataset_name in args.datasets:
            args.eval_noise_mode = (
                "flip" if dataset_name in {"Cora", "CiteSeer"} else "noise"
            )

            try:
                run_one_dataset(dataset_name, args, scenario_name=scenario["name"])
                print(f"FINISHED DATASET: {dataset_name} " f"[{scenario['name']}]")
            except Exception as exc:
                print(f"FAILED DATASET: {dataset_name} " f"[{scenario['name']}]: {exc}")

        print(f"FINISHED SCENARIO: {scenario['name']}")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    run_suite()
