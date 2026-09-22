# GNN Robustness Experiments

This repository contains the code for the thesis **Robustness Analysis of Graph Neural Networks under Structural and Feature Distribution Shifts**.

The experiments study node-classification performance under feature corruption, edge dropout, and combined feature and structural perturbations.

> **Execution note:** The complete thesis experiment suite was executed on Kaggle GPU-enabled environments. It combines 22 scenarios, 6 datasets, 3 models, 5 perturbation levels, and 10 repeated runs, so it is substantially more expensive than a normal local test run. The repository includes a small local verification configuration, but the full thesis workflow should be run in Kaggle or another suitable compute environment.

## Project Structure

```text
code/
  main.py                Experiment entry point
  scenarios.py           Experiment scenario definitions
  experiment_runner.py   Dataset loading, orchestration, and result saving
  training.py            GCN and feature-only model training
  evaluation.py          Perturbation evaluation and centrality scores
  model.py               GCN, MLP, and Logistic Regression models
  graph_features.py      Graph centrality calculations
  utils.py               Feature and edge perturbation utilities

requirements.txt          Python package requirements
results/                  Generated experiment outputs
data/                     Downloaded PyTorch Geometric datasets
```

## Environment

The experiments use Python 3.10 and the packages listed in `requirements.txt`.

Main dependencies:

- Python 3.10
- PyTorch
- PyTorch Geometric
- NumPy
- Matplotlib
- scikit-learn
- igraph

To create a Conda environment and install the required packages:

```powershell
conda create -n gnn_robustness python=3.10 -y
conda activate gnn_robustness
pip install -r requirements.txt
```

You can use any environment name. If an environment is already available, install the dependencies directly with `pip install -r requirements.txt`.

## Running the Experiments

From the repository root:

```powershell
python code/main.py
```

The entry point iterates through the scenarios in `code/scenarios.py`. For each scenario, it iterates through the configured datasets in `BaseArgs.datasets` in `code/main.py`.

The complete suite uses the configured training epochs, runs, and perturbation levels. The scenario descriptions are printed before each experiment.

A small configuration can be used locally to verify the workflow before running the full suite.

For a quick local verification, temporarily change `BaseArgs` in `code/main.py` to:

```python
datasets = ["CiteSeer"]
epochs = 2
runs = 1
noise_levels = [0.0, 0.2]
```

Then run:

```powershell
python code/main.py
```

Restore the full `BaseArgs` values before running the thesis experiment suite.

## Dataset Loading

Dataset selection is implemented in `code/experiment_runner.py`, in the `run_one_dataset()` function. The dataset name passed from `main.py` selects the corresponding PyTorch Geometric dataset class:

| Repository name | PyTorch Geometric dataset | Source type |
|---|---|---|
| `Cora` | `Planetoid(name="Cora")` | Citation network |
| `CiteSeer` | `Planetoid(name="CiteSeer")` | Citation network |
| `Pubmed` | `Planetoid(name="Pubmed")` | Citation network |
| `CS` | `Coauthor(name="CS")` | Co-authorship network |
| `Physics` | `Coauthor(name="Physics")` | Co-authorship network |
| `Photo` | `Amazon(name="Photo")` | Product co-purchasing network |

The datasets are downloaded and processed automatically by PyTorch Geometric when they are not already present under `data/`.

## Models

Three models are evaluated:

- **GCN:** the primary graph-based model. It uses node features and graph edges and can receive training-time robustness augmentations.
- **MLP:** a feature-only neural-network baseline. It does not use graph edges.
- **Logistic Regression:** a feature-only classical baseline. It does not use graph edges.

Training-time robustness interventions are applied to the GCN. MLP and Logistic Regression remain clean feature-only reference baselines.

## Feature Perturbations

Feature perturbation is selected according to the dataset feature representation:

- Cora and CiteSeer use feature flipping.
- PubMed, Coauthor-CS, Coauthor-Physics, and Amazon-Photo use additive Gaussian noise.

The evaluation levels are:

```text
0.0, 0.2, 0.4, 0.6, 0.9
```

For Gaussian noise, the level is the standard scale used by the implementation. For feature flipping, it is the probability used to select feature entries.

## Structural Perturbations

The code supports:

- Random edge dropout
- Degree-based probabilistic edge removal
- PageRank-based probabilistic edge removal
- Betweenness-based probabilistic edge removal

Undirected graph centrality is calculated using one logical edge per node pair. PyTorch Geometric stores undirected relationships in both directions, so the centrality graph removes those duplicate directions before calculating scores.

## Scenarios

Scenario definitions are stored in `code/scenarios.py`. The current suite includes:

- Clean baseline evaluation
- Feature corruption evaluation
- Edge-dropout evaluation
- Combined feature and edge corruption
- GCN feature augmentation
- GCN edge-dropout training
- GCN feature and edge augmentation
- GCN degree-based edge sampling
- GCN PageRank-based edge sampling
- GCN betweenness-based edge sampling

Each scenario name describes the training intervention and evaluation condition. For example:

```text
Baseline__FeatureCorruption
GCNFeatureAugmentation__FeatureCorruption
GCNPageRankBasedEdgeSampling__FeatureAndEdgeCorruption
```

## Results

Results are generated automatically under:

```text
results/<scenario_name>/<dataset_name>/
```

Each dataset experiment produces:

```text
results.json
summary.csv
accuracy_plot.png
```

The JSON file contains configuration metadata, dataset information, perturbation levels, individual run values, means, and standard deviations. The CSV file contains a table-friendly summary.

## Reproducibility

The code sets seeds for Python, NumPy, and PyTorch before dataset splitting and repeated runs. The default experiment configuration uses:

```text
Hidden dimension: 64
Dropout: 0.5
Learning rate: 0.001
Epochs: 400
Runs: 10
Training augmentation level: 0.1
```

The evaluation split uses `RandomNodeSplit` with 20 training nodes per class, 500 validation nodes, and 1,000 test nodes.

## Optional Evaluation Shield

The evaluation module contains an optional feature-similarity graph filtering method called the evaluation shield. It is disabled in the current thesis scenarios:

```python
use_eval_shield = False
```

It is retained as an optional extension and does not affect the default experiments.
