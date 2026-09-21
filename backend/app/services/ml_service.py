"""
ML Service - graph-based fragility scoring.

The scorer uses lightweight graph features and iterative dependency-risk
propagation. It does not require PyTorch or PyTorch Geometric.

Public API:
    compute_fragility_scores(...)
        Backward-compatible function returning node_id -> score.

    compute_fragility_analysis(...)
        Extended function returning scores plus explainability data.
"""

import logging
from typing import Dict, List, Any, Set, Tuple

logger = logging.getLogger(__name__)


def _validate_parameters(iterations: int, damping: float) -> None:
    """Validate scoring parameters before running the algorithm."""
    if not isinstance(iterations, int) or isinstance(iterations, bool):
        raise ValueError("iterations must be an integer")

    if iterations < 1:
        raise ValueError("iterations must be at least 1")

    if not isinstance(damping, (int, float)) or isinstance(damping, bool):
        raise ValueError("damping must be a number")

    if not 0.0 <= float(damping) <= 1.0:
        raise ValueError("damping must be between 0.0 and 1.0")


def _build_graph(
    nodes: List[Dict[str, Any]],
    edges: List[Dict[str, Any]],
) -> Tuple[
    List[str],
    Dict[str, Set[str]],
    Dict[str, Set[str]],
]:
    """
    Build a duplicate-safe directed graph.

    outgoing[source] = nodes that source depends on
    incoming[target] = nodes that depend on target

    Sets ensure duplicate edges are counted only once.
    """
    node_ids = [node["id"] for node in nodes]

    outgoing: Dict[str, Set[str]] = {
        node_id: set() for node_id in node_ids
    }

    incoming: Dict[str, Set[str]] = {
        node_id: set() for node_id in node_ids
    }

    valid_node_ids = set(node_ids)

    for edge in edges:
        source = edge.get("source")
        target = edge.get("target")

        if source not in valid_node_ids or target not in valid_node_ids:
            continue

        if source == target:
            continue

        outgoing[source].add(target)
        incoming[target].add(source)

    return node_ids, outgoing, incoming


def _normalise(value: int, maximum: int) -> float:
    """Normalise a graph metric to the range 0.0-1.0."""
    if maximum <= 0:
        return 0.0

    return min(value / maximum, 1.0)


def _build_structural_metrics(
    nodes: List[Dict[str, Any]],
    node_ids: List[str],
    outgoing: Dict[str, Set[str]],
    incoming: Dict[str, Set[str]],
) -> Dict[str, Dict[str, float]]:
    """
    Calculate fan-in, fan-out and connectivity for every node.

    All returned metrics are normalised to 0.0-1.0.
    """
    raw_fan_in = {
        node_id: len(incoming[node_id])
        for node_id in node_ids
    }

    raw_fan_out = {
        node_id: len(outgoing[node_id])
        for node_id in node_ids
    }

    raw_connectivity = {
        node_id: len(outgoing[node_id] | incoming[node_id])
        for node_id in node_ids
    }

    max_fan_in = max(raw_fan_in.values(), default=0)
    max_fan_out = max(raw_fan_out.values(), default=0)
    max_connectivity = max(raw_connectivity.values(), default=0)

    node_types = {
        node.get("id"): node.get("type", "")
        for node in nodes
    }

    metrics: Dict[str, Dict[str, float]] = {}

    for node_id in node_ids:
        fan_in = _normalise(raw_fan_in[node_id], max_fan_in)
        fan_out = _normalise(raw_fan_out[node_id], max_fan_out)
        connectivity = _normalise(
            raw_connectivity[node_id],
            max_connectivity,
        )

        # Functions receive a small weight because they are executable
        # units and are the primary targets of function-level analysis.
        type_weight = 1.2 if node_types.get(node_id) == "function" else 1.0

        # Structural risk combines three distinct graph signals.
        structural_risk = (
            0.40 * fan_in
            + 0.30 * fan_out
            + 0.30 * connectivity
        )

        structural_risk = min(
            structural_risk * type_weight,
            1.0,
        )

        metrics[node_id] = {
            "fan_in": fan_in,
            "fan_out": fan_out,
            "connectivity": connectivity,
            "structural_risk": structural_risk,
        }

    return metrics


def _propagate_risk(
    node_ids: List[str],
    outgoing: Dict[str, Set[str]],
    base_risk: Dict[str, float],
    iterations: int,
    damping: float,
) -> Dict[str, float]:
    """
    Propagate risk through dependency relationships.

    If A -> B, A depends on B, so some of B's risk can propagate to A.
    The original structural risk remains part of every iteration.
    """
    risk = dict(base_risk)

    for _ in range(iterations):
        new_risk: Dict[str, float] = {}

        for node_id in node_ids:
            dependencies = outgoing[node_id]

            if dependencies:
                dependency_risk = sum(
                    risk[dependency]
                    for dependency in dependencies
                ) / len(dependencies)
            else:
                dependency_risk = 0.0

            propagated = (
                (1.0 - damping) * base_risk[node_id]
                + damping * dependency_risk
            )

            new_risk[node_id] = min(max(propagated, 0.0), 1.0)

        risk = new_risk

    return risk


def _build_explanation(
    metrics: Dict[str, float],
    propagated_risk: float,
) -> List[str]:
    """Create simple human-readable reasons for a node's risk."""
    reasons: List[str] = []

    if metrics["fan_in"] >= 0.66:
        reasons.append("high fan-in")
    elif metrics["fan_in"] >= 0.33:
        reasons.append("moderate fan-in")

    if metrics["fan_out"] >= 0.66:
        reasons.append("high fan-out")
    elif metrics["fan_out"] >= 0.33:
        reasons.append("moderate fan-out")

    if metrics["connectivity"] >= 0.66:
        reasons.append("high connectivity")
    elif metrics["connectivity"] >= 0.33:
        reasons.append("moderate connectivity")

    if propagated_risk > metrics["structural_risk"] + 0.05:
        reasons.append("elevated dependency propagation")

    if not reasons:
        reasons.append("low structural connectivity")

    return reasons


def compute_fragility_analysis(
    nodes: List[Dict[str, Any]],
    edges: List[Dict[str, Any]],
    iterations: int = 3,
    damping: float = 0.85,
) -> Dict[str, Dict[str, Any]]:
    """
    Compute fragility scores and structural explanations.

    Returns:
        {
            node_id: {
                "score": float,
                "fan_in": float,
                "fan_out": float,
                "connectivity": float,
                "structural_risk": float,
                "propagated_risk": float,
                "reasons": [...]
            }
        }

    Scores are always bounded between 0 and 100.
    """
    _validate_parameters(iterations, damping)

    if not nodes:
        return {}

    node_ids, outgoing, incoming = _build_graph(nodes, edges)

    metrics = _build_structural_metrics(
        nodes,
        node_ids,
        outgoing,
        incoming,
    )

    base_risk = {
        node_id: metrics[node_id]["structural_risk"]
        for node_id in node_ids
    }

    propagated_risk = _propagate_risk(
        node_ids,
        outgoing,
        base_risk,
        iterations,
        float(damping),
    )

    results: Dict[str, Dict[str, Any]] = {}

    for node_id in node_ids:
        node_metrics = metrics[node_id]

        # Keep structural characteristics as the dominant signal while
        # allowing dependency propagation to influence the final score.
        final_risk = (
            0.70 * node_metrics["structural_risk"]
            + 0.30 * propagated_risk[node_id]
        )

        final_risk = min(max(final_risk, 0.0), 1.0)

        score = round(final_risk * 100.0, 1)

        results[node_id] = {
            "score": score,
            "fan_in": round(node_metrics["fan_in"], 4),
            "fan_out": round(node_metrics["fan_out"], 4),
            "connectivity": round(node_metrics["connectivity"], 4),
            "structural_risk": round(
                node_metrics["structural_risk"],
                4,
            ),
            "propagated_risk": round(
                propagated_risk[node_id],
                4,
            ),
            "reasons": _build_explanation(
                node_metrics,
                propagated_risk[node_id],
            ),
        }

    return results


def compute_fragility_scores(
    nodes: List[Dict[str, Any]],
    edges: List[Dict[str, Any]],
    iterations: int = 3,
    damping: float = 0.85,
) -> Dict[str, float]:
    """
    Backward-compatible public API.

    Returns:
        node_id -> fragility score between 0 and 100
    """
    analysis = compute_fragility_analysis(
        nodes,
        edges,
        iterations=iterations,
        damping=damping,
    )

    return {
        node_id: details["score"]
        for node_id, details in analysis.items()
    }