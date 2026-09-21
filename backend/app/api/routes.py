"""
API routes for FragilityGraph.

Endpoints:

    GET  /api/v1/health
    GET  /api/v1/file_tree
    GET  /api/v1/line_risks
    GET  /api/v1/graph_data
    POST /api/v1/analyze
    POST /api/v1/analyze_focused
    POST /api/v1/impact_analysis
    POST /api/v1/explain_node
"""

import os
import logging
from typing import Optional

from fastapi import APIRouter, Query, HTTPException
from pydantic import BaseModel

from app.config import settings
from app.models.schemas import (
    FileNode,
    GraphData,
    GraphNode,
    GraphEdge,
    LineRiskResponse,
    LineRisk,
)
from app.services import line_analyzer, ml_service, bedrock_service
from app.graph.neo4j_adapter import Neo4jAdapter


logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1")


# Directories to ignore when building the file tree
IGNORED_DIRS = {
    "node_modules",
    "venv",
    ".venv",
    ".git",
    "__pycache__",
    ".vscode",
    ".idea",
    "dist",
    "build",
    ".mypy_cache",
    ".pytest_cache",
    "FinalDocs",
    ".gemini",
}


# ── Request Models ─────────────────────────────────────────

class ImpactRequest(BaseModel):
    file_path: str
    change_description: str


class ExplainRequest(BaseModel):
    node_id: str
    label: str
    file_path: str
    fragility: float = 0.0


# ── Health ─────────────────────────────────────────────────

@router.get("/health")
def health_check():
    return {"status": "ok"}


# ── File tree ──────────────────────────────────────────────

@router.get("/file_tree")
def get_file_tree(root: Optional[str] = None):
    """Return a nested JSON tree of the project directory."""
    scan_root = root or settings.PROJECT_ROOT

    if not os.path.isdir(scan_root):
        raise HTTPException(
            status_code=400,
            detail=f"Not a directory: {scan_root}",
        )

    tree = _build_tree(scan_root)
    return tree


def _build_tree(path: str) -> dict:
    name = os.path.basename(path)

    if os.path.isfile(path):
        return {
            "name": name,
            "path": path.replace("\\", "/"),
            "type": "file",
        }

    children = []

    try:
        entries = sorted(os.listdir(path))
    except PermissionError:
        entries = []

    for entry in entries:
        if entry in IGNORED_DIRS or entry.startswith("."):
            continue

        full = os.path.join(path, entry)
        children.append(_build_tree(full))

    return {
        "name": name,
        "path": path.replace("\\", "/"),
        "type": "directory",
        "children": children,
    }


# ── Line risks ─────────────────────────────────────────────

@router.get("/line_risks", response_model=LineRiskResponse)
def get_line_risks(
    file_path: str = Query(..., description="Absolute path to file"),
    focus_path: Optional[str] = Query(
        None,
        description="Focused root file path",
    ),
):
    """Parse a file and return per-line risk annotations."""

    if not os.path.isfile(file_path):
        raise HTTPException(
            status_code=404,
            detail="File not found",
        )

    with open(
        file_path,
        "r",
        encoding="utf-8",
        errors="replace",
    ) as f:
        content = f.read()

    adapter = Neo4jAdapter.get_instance()

    fragile_map = {}

    for node in adapter.get_all_nodes():
        if node.get("fragility", 0) > 0:
            fragile_map[node.get("label", "")] = node["fragility"]

    risks = line_analyzer.compute_line_risks(
        content,
        file_path,
        fragile_map,
        focus_path=focus_path or "",
    )

    return LineRiskResponse(
        file_path=file_path,
        content=content,
        lines=[LineRisk(**r) for r in risks],
    )


# ── Graph data ─────────────────────────────────────────────

@router.get("/graph_data", response_model=GraphData)
def get_graph_data():
    """Return the full dependency graph for the frontend visualisation."""

    adapter = Neo4jAdapter.get_instance()

    raw_nodes = adapter.get_all_nodes()
    raw_edges = adapter.get_all_edges()

    nodes = [
        GraphNode(
            id=n.get("id", ""),
            label=n.get("label", ""),
            type=n.get("type", "function"),
            file_path=n.get("file_path", ""),
            line_number=n.get("line_number", 0),
            fragility=n.get("fragility", 0.0),
        )
        for n in raw_nodes
    ]

    edges = [
        GraphEdge(
            source=e.get("source", ""),
            target=e.get("target", ""),
            relationship=e.get("relationship", "CALLS"),
        )
        for e in raw_edges
    ]

    return GraphData(
        nodes=nodes,
        edges=edges,
    )


# ── Analyze a single file ──────────────────────────────────

@router.post("/analyze")
def analyze_file(file_path: str = Query(...)):
    """
    Parse a Python file, update the graph, compute fragility,
    and return the results.
    """

    if not os.path.isfile(file_path):
        raise HTTPException(
            status_code=404,
            detail="File not found",
        )

    with open(
        file_path,
        "r",
        encoding="utf-8",
        errors="replace",
    ) as f:
        content = f.read()

    nodes, edges = line_analyzer.extract_graph_elements(
        content,
        file_path,
    )

    adapter = Neo4jAdapter.get_instance()
    adapter.bulk_upsert(nodes, edges)

    # Existing score API remains unchanged.
    scores = ml_service.compute_fragility_scores(
        nodes,
        edges,
    )

    # New extended analysis provides explainability information.
    fragility_analysis = ml_service.compute_fragility_analysis(
        nodes,
        edges,
    )

    for node in nodes:
        node_id = node["id"]

        score = scores.get(node_id, 0.0)
        node["fragility"] = score

        # Keep the detailed structural explanation with the node.
        analysis = fragility_analysis.get(node_id)

        if analysis is not None:
            node["fragility_analysis"] = analysis

        adapter.update_fragility_score(
            node_id,
            score,
        )

    fragile_map = {
        n["label"]: n["fragility"]
        for n in nodes
        if n["fragility"] > 0
    }

    line_risks = line_analyzer.compute_line_risks(
        content,
        file_path,
        fragile_map,
    )

    return {
        "file_path": file_path,
        "nodes": nodes,
        "edges": edges,
        "line_risks": line_risks,
    }


# ── Analyze focused (file-scoped graph + AI summary) ──────

@router.post("/analyze_focused")
def analyze_focused(file_path: str = Query(...)):
    """
    Focused analysis: analyse one file, compute fragility,
    and return file-level graph + line risks + AI file summary.
    """

    if not os.path.isfile(file_path):
        raise HTTPException(
            status_code=404,
            detail="File not found",
        )

    with open(
        file_path,
        "r",
        encoding="utf-8",
        errors="replace",
    ) as f:
        content = f.read()

    # ── 1. File-level graph for visualization ──

    graph_nodes, graph_edges = line_analyzer.extract_graph_elements(
        content,
        file_path,
    )

    # Push file-level graph to Neo4j
    adapter = Neo4jAdapter.get_instance()

    adapter.bulk_upsert(
        graph_nodes,
        graph_edges,
    )

    # ── 2. Function-level analysis for ML scoring & line risks ──

    analysis = line_analyzer.analyse_file(
        content,
        file_path,
    )

    func_nodes = []
    func_edges = []
    defined_names = set()

    for defn in analysis["definitions"]:
        node_id = f"{file_path}::{defn['name']}"

        func_nodes.append(
            {
                "id": node_id,
                "label": defn["name"],
                "type": defn["type"],
                "file_path": file_path,
                "line_number": defn["line_start"],
                "fragility": 0.0,
            }
        )

        defined_names.add(defn["name"])

    for call in analysis["calls"]:
        callee = call["name"].split(".")[-1]

        if callee in defined_names:
            caller_id = None

            for defn in analysis["definitions"]:
                if (
                    defn["line_start"]
                    <= call["line"]
                    <= defn["line_end"]
                ):
                    caller_id = f"{file_path}::{defn['name']}"
                    break

            if caller_id:
                target_id = f"{file_path}::{callee}"

                if caller_id != target_id:
                    func_edges.append(
                        {
                            "source": caller_id,
                            "target": target_id,
                            "relationship": "CALLS",
                        }
                    )

    # ── 3. Compute fragility scores ──

    # A. Scale-level scores for the neighbourhood graph.
    file_scores = ml_service.compute_fragility_scores(
        graph_nodes,
        graph_edges,
    )

    file_analysis = ml_service.compute_fragility_analysis(
        graph_nodes,
        graph_edges,
    )

    for gn in graph_nodes:
        node_id = gn["id"]

        gn["fragility"] = file_scores.get(
            node_id,
            0.0,
        )

        analysis_data = file_analysis.get(node_id)

        if analysis_data is not None:
            gn["fragility_analysis"] = analysis_data

    # B. Detailed function-level scores for the root file.
    func_scores = ml_service.compute_fragility_scores(
        func_nodes,
        func_edges,
    )

    func_analysis = ml_service.compute_fragility_analysis(
        func_nodes,
        func_edges,
    )

    max_root_fragility = 0.0

    for node in func_nodes:
        node_id = node["id"]

        score = func_scores.get(
            node_id,
            0.0,
        )

        node["fragility"] = score

        max_root_fragility = max(
            max_root_fragility,
            score,
        )

        analysis_data = func_analysis.get(node_id)

        if analysis_data is not None:
            node["fragility_analysis"] = analysis_data

    # Root file node should use the maximum internal fragility.
    norm_root = os.path.abspath(file_path).replace(
        "\\",
        "/",
    )

    for gn in graph_nodes:
        if gn["id"] == norm_root:
            gn["fragility"] = max_root_fragility
            break

    # ── 4. Line risks ──

    fragile_map = {
        n["label"]: n["fragility"]
        for n in func_nodes
        if n["fragility"] > 0
    }

    line_risks = line_analyzer.compute_line_risks(
        content,
        file_path,
        fragile_map,
        focus_path=file_path,
    )

    # ── 5. AI file summary ──

    summary = bedrock_service.summarize_file(
        file_path=file_path,
        content=content,
        node_count=len(func_nodes),
        edge_count=len(func_edges),
        max_fragility=max_root_fragility,
    )

    return {
        "file_path": file_path,
        "nodes": graph_nodes,
        "edges": graph_edges,
        "line_risks": line_risks,
        "summary": summary,
        "max_fragility": max_root_fragility,

        # New explainability information.
        "fragility_analysis": func_analysis,
    }


# ── Change impact analysis ────────────────────────────────

@router.post("/impact_analysis")
def impact_analysis(req: ImpactRequest):
    """
    Given a file and a change description, identify which functions
    would be affected by the proposed change.
    """

    if not os.path.isfile(req.file_path):
        raise HTTPException(
            status_code=404,
            detail="File not found",
        )

    with open(
        req.file_path,
        "r",
        encoding="utf-8",
        errors="replace",
    ) as f:
        content = f.read()

    # Get existing definitions
    analysis = line_analyzer.analyse_file(
        content,
        req.file_path,
    )

    function_names = [
        d["name"]
        for d in analysis["definitions"]
    ]

    # Ask Bedrock for impact analysis
    affected = bedrock_service.analyze_change_impact(
        file_path=req.file_path,
        content=content,
        change_description=req.change_description,
        function_names=function_names,
    )

    # Build affected node IDs
    affected_ids = [
        f"{req.file_path}::{name}"
        for name in affected
    ]

    return {
        "file_path": req.file_path,
        "change_description": req.change_description,
        "affected_functions": affected,
        "affected_node_ids": affected_ids,
        "total_functions": len(function_names),
    }


# ── Single node AI explanation ─────────────────────────────

@router.post("/explain_node")
def explain_node(req: ExplainRequest):
    """Get an AI explanation for a specific graph node."""

    code_snippet = ""

    if os.path.isfile(req.file_path):
        with open(
            req.file_path,
            "r",
            encoding="utf-8",
            errors="replace",
        ) as f:
            content = f.read()

        # Find the function/class definition.
        analysis = line_analyzer.analyse_file(
            content,
            req.file_path,
        )

        for defn in analysis["definitions"]:
            if defn["name"] == req.label:
                lines = content.splitlines()

                start = max(
                    0,
                    defn["line_start"] - 1,
                )

                end = min(
                    len(lines),
                    defn["line_end"],
                )

                code_snippet = "\n".join(
                    lines[start:end]
                )

                break

    # Get dependencies from graph
    adapter = Neo4jAdapter.get_instance()

    all_edges = adapter.get_all_edges()

    dep_names = [
        e.get("target", "").split("::")[-1]
        for e in all_edges
        if e.get("source") == req.node_id
    ]

    explanation = bedrock_service.explain_fragility(
        function_name=req.label,
        code_snippet=(
            code_snippet
            if code_snippet
            else f"# {req.label}"
        ),
        fragility_score=req.fragility,
        dependencies=dep_names,
    )

    return {
        "node_id": req.node_id,
        "explanation": explanation,
        "dependencies": dep_names,
    }