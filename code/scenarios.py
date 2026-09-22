"""Explicit experiment configurations used by the thesis workflow."""

SCENARIOS = [
    {
        "name": "Baseline__CleanData",
        "feat_mode": None,
        "edge_mode": None,
        "eval_edge_mode": None,
        "eval_feature_noise": False,
    },
    {
        "name": "Baseline__FeatureCorruption",
        "feat_mode": None,
        "edge_mode": None,
        "eval_edge_mode": None,
        "eval_feature_noise": True,
    },
    {
        "name": "Baseline__EdgeDropout",
        "feat_mode": None,
        "edge_mode": None,
        "eval_edge_mode": "dropout",
        "eval_feature_noise": False,
    },
    {
        "name": "Baseline__FeatureAndEdgeCorruption",
        "feat_mode": None,
        "edge_mode": None,
        "eval_edge_mode": "dropout",
        "eval_feature_noise": True,
    },
    {
        "name": "GCNFeatureAugmentation__FeatureCorruption",
        "feat_mode": "noise",
        "edge_mode": None,
        "eval_edge_mode": None,
        "eval_feature_noise": True,
    },
    {
        "name": "GCNFeatureAugmentation__EdgeDropout",
        "feat_mode": "noise",
        "edge_mode": None,
        "eval_edge_mode": "dropout",
        "eval_feature_noise": False,
    },
    {
        "name": "GCNFeatureAugmentation__FeatureAndEdgeCorruption",
        "feat_mode": "noise",
        "edge_mode": None,
        "eval_edge_mode": "dropout",
        "eval_feature_noise": True,
    },
    {
        "name": "GCNEdgeDropoutTraining__FeatureCorruption",
        "feat_mode": None,
        "edge_mode": "dropout",
        "eval_edge_mode": None,
        "eval_feature_noise": True,
    },
    {
        "name": "GCNEdgeDropoutTraining__EdgeDropout",
        "feat_mode": None,
        "edge_mode": "dropout",
        "eval_edge_mode": "dropout",
        "eval_feature_noise": False,
    },
    {
        "name": "GCNEdgeDropoutTraining__FeatureAndEdgeCorruption",
        "feat_mode": None,
        "edge_mode": "dropout",
        "eval_edge_mode": "dropout",
        "eval_feature_noise": True,
    },
    {
        "name": "GCNFeatureAndEdgeAugmentation__FeatureAndEdgeCorruption",
        "feat_mode": "noise",
        "edge_mode": "dropout",
        "eval_edge_mode": "dropout",
        "eval_feature_noise": True,
    },
    {
        "name": "GCNFeatureAndEdgeAugmentation__FeatureCorruption",
        "feat_mode": "noise",
        "edge_mode": "dropout",
        "eval_edge_mode": None,
        "eval_feature_noise": True,
    },
    {
        "name": "GCNFeatureAndEdgeAugmentation__EdgeDropout",
        "feat_mode": "noise",
        "edge_mode": "dropout",
        "eval_edge_mode": "dropout",
        "eval_feature_noise": False,
    },
    {
        "name": "GCNDegreeBasedEdgeSampling__EdgeDropout",
        "feat_mode": None,
        "edge_mode": "degree_prob",
        "eval_edge_mode": "dropout",
        "eval_feature_noise": False,
    },
    {
        "name": "GCNDegreeBasedEdgeSampling__FeatureCorruption",
        "feat_mode": None,
        "edge_mode": "degree_prob",
        "eval_edge_mode": None,
        "eval_feature_noise": True,
    },
    {
        "name": "GCNDegreeBasedEdgeSampling__FeatureAndEdgeCorruption",
        "feat_mode": None,
        "edge_mode": "degree_prob",
        "eval_edge_mode": "dropout",
        "eval_feature_noise": True,
    },
    {
        "name": "GCNPageRankBasedEdgeSampling__EdgeDropout",
        "feat_mode": None,
        "edge_mode": "pagerank_prob",
        "eval_edge_mode": "dropout",
        "eval_feature_noise": False,
    },
    {
        "name": "GCNPageRankBasedEdgeSampling__FeatureCorruption",
        "feat_mode": None,
        "edge_mode": "pagerank_prob",
        "eval_edge_mode": None,
        "eval_feature_noise": True,
    },
    {
        "name": "GCNPageRankBasedEdgeSampling__FeatureAndEdgeCorruption",
        "feat_mode": None,
        "edge_mode": "pagerank_prob",
        "eval_edge_mode": "dropout",
        "eval_feature_noise": True,
    },
    {
        "name": "GCNBetweennessBasedEdgeSampling__EdgeDropout",
        "feat_mode": None,
        "edge_mode": "bw_prob",
        "eval_edge_mode": "dropout",
        "eval_feature_noise": False,
    },
    {
        "name": "GCNBetweennessBasedEdgeSampling__FeatureCorruption",
        "feat_mode": None,
        "edge_mode": "bw_prob",
        "eval_edge_mode": None,
        "eval_feature_noise": True,
    },
    {
        "name": "GCNBetweennessBasedEdgeSampling__FeatureAndEdgeCorruption",
        "feat_mode": None,
        "edge_mode": "bw_prob",
        "eval_edge_mode": "dropout",
        "eval_feature_noise": True,
    },
]


def describe_scenario(scenario):
    train_parts = []
    if scenario["feat_mode"] == "noise":
        train_parts.append("10% Gaussian feature augmentation")
    if scenario["edge_mode"] == "dropout":
        train_parts.append("10% random edge dropout")
    if scenario["edge_mode"] == "degree_prob":
        train_parts.append("degree-weighted edge sampling")
    if scenario["edge_mode"] == "pagerank_prob":
        train_parts.append("PageRank-weighted edge sampling")
    if scenario["edge_mode"] == "bw_prob":
        train_parts.append("betweenness-weighted edge sampling")

    training = (
        "clean features and the original graph"
        if not train_parts
        else "clean data with " + " and ".join(train_parts)
    )

    evaluation_parts = []
    if scenario["eval_feature_noise"]:
        evaluation_parts.append(
            "feature corruption (flipping for Cora/CiteSeer and Gaussian noise otherwise)"
        )
    if scenario["eval_edge_mode"] == "dropout":
        evaluation_parts.append("edge dropout")

    evaluation = (
        "clean features and the original graph"
        if not evaluation_parts
        else " and ".join(evaluation_parts)
    )

    return f"Train on {training}; evaluate under {evaluation}."
