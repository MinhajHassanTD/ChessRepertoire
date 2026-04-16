"""
3D graph visualizer for the positions database.
Each node = chess position, labeled by the move line to reach it.
Color coded: blue = White to move, orange = Black to move.
Run: python visualize/graph_3d.py
Opens an interactive HTML file in browser.
"""

import os
import sys
import pickle
import webbrowser
from collections import deque

import numpy as np
import plotly.graph_objects as go

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from src.graph import load_graph, build_graph

# ── Config ────────────────────────────────────────────────────────────────────

GRAPH_PKL  = "data/graph_train.pkl"
SNAPSHOT_DB = "data/snapshot.db"
OUTPUT_HTML = "visualize/graph_3d.html"

MAX_DEPTH   = 10   # cap depth shown (reduce clutter)
MIN_GAMES   = 0   # only show nodes with total_games >= this

WHITE_COLOR = "#5B9BD5"   # blue  = white to move
BLACK_COLOR = "#FF6B35"   # orange = black to move
EDGE_COLOR  = "rgba(150,150,150,0.25)"


# ── Step 1: load graph ────────────────────────────────────────────────────────

def load(graph_pkl=GRAPH_PKL, db=SNAPSHOT_DB):
    if os.path.exists(graph_pkl):
        print(f"Loading graph from {graph_pkl} ...")
        return load_graph(graph_pkl)
    print(f"PKL not found, building from {db} ...")
    g = build_graph(db, "train")
    return g


# ── Step 2: BFS to assign move-line labels ────────────────────────────────────

def build_move_lines(graph):
    """
    BFS from root. For each FEN, record the shortest SAN move sequence.
    Returns {fen: move_line_string}.
    """
    root  = graph["root_fen"]
    nodes = graph["nodes"]

    line  = {root: ""}          # root = start position, empty line
    queue = deque([root])
    visited = {root}

    while queue:
        fen = queue.popleft()
        node = nodes.get(fen)
        if node is None:
            continue
        for uci, child in node["children"].items():
            child_fen = child["child_fen"]
            if child_fen not in visited and child_fen in nodes:
                visited.add(child_fen)
                san = child["move_san"]
                parent_line = line[fen]
                line[child_fen] = (parent_line + " " + san).strip()
                queue.append(child_fen)

    return line


# ── Step 3: 3D layout  ────────────────────────────────────────────────────────

def compute_positions_3d(graph, move_lines, max_depth=MAX_DEPTH):
    """
    Z = ply_depth.
    X/Y = nodes at each depth spread in a circle.
    Returns {fen: (x, y, z)}.
    """
    nodes = graph["nodes"]

    # Group fens by depth
    depth_groups: dict[int, list] = {}
    for fen, node in nodes.items():
        d = node["ply_depth"]
        if d > max_depth:
            continue
        depth_groups.setdefault(d, []).append(fen)

    pos3d = {}

    for depth, fens in depth_groups.items():
        n = len(fens)
        z = float(depth)
        radius = max(1.0, n / (2 * np.pi) * 0.8)   # ring radius scales with count

        for i, fen in enumerate(fens):
            angle = 2 * np.pi * i / n
            x = radius * np.cos(angle)
            y = radius * np.sin(angle)
            pos3d[fen] = (x, y, z)

    return pos3d


# ── Step 4: build plotly traces ───────────────────────────────────────────────

def make_figure(graph, move_lines, pos3d, min_games=MIN_GAMES):
    nodes = graph["nodes"]

    # Separate white/black to move
    white_fens, black_fens = [], []
    for fen, (x, y, z) in pos3d.items():
        node = nodes.get(fen)
        if node is None:
            continue
        if node["total_games"] < min_games:
            continue
        if node["turn"] == "white":
            white_fens.append(fen)
        else:
            black_fens.append(fen)

    def make_node_trace(fens, color, name):
        xs, ys, zs, texts, sizes = [], [], [], [], []
        for fen in fens:
            node = nodes[fen]
            x, y, z = pos3d[fen]
            xs.append(x); ys.append(y); zs.append(z)
            line = move_lines.get(fen, "?")
            label = line if line else "(start)"
            g  = node["total_games"]
            sc = round(node["score_raw"], 3)
            texts.append(
                f"<b>{label}</b><br>"
                f"Depth: {node['ply_depth']}<br>"
                f"Games: {g:,}<br>"
                f"Score: {sc}"
            )
            # Node size: log-scaled by games
            sizes.append(max(4, min(18, 3 + np.log10(max(g, 1)) * 3)))

        return go.Scatter3d(
            x=xs, y=ys, z=zs,
            mode="markers",
            name=name,
            marker=dict(size=sizes, color=color, opacity=0.85,
                        line=dict(width=0.3, color="white")),
            hovertext=texts,
            hoverinfo="text",
        )

    # Edge trace
    ex, ey, ez = [], [], []
    for fen, node in nodes.items():
        if fen not in pos3d:
            continue
        x1, y1, z1 = pos3d[fen]
        for uci, child in node["children"].items():
            child_fen = child["child_fen"]
            if child_fen not in pos3d:
                continue
            x2, y2, z2 = pos3d[child_fen]
            ex += [x1, x2, None]
            ey += [y1, y2, None]
            ez += [z1, z2, None]

    edge_trace = go.Scatter3d(
        x=ex, y=ey, z=ez,
        mode="lines",
        name="edges",
        line=dict(color=EDGE_COLOR, width=0.8),
        hoverinfo="none",
    )

    fig = go.Figure(data=[
        edge_trace,
        make_node_trace(white_fens, WHITE_COLOR, "White to move"),
        make_node_trace(black_fens, BLACK_COLOR, "Black to move"),
    ])

    total_nodes = len(white_fens) + len(black_fens)
    fig.update_layout(
        title=dict(
            text=f"Chess Opening Tree — {total_nodes} positions (depth 0-{MAX_DEPTH})",
            font=dict(size=16),
        ),
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(title="Ply depth", showgrid=True, gridcolor="rgba(200,200,200,0.3)"),
            bgcolor="rgb(15,15,25)",
            camera=dict(eye=dict(x=1.6, y=1.6, z=0.8)),
        ),
        paper_bgcolor="rgb(15,15,25)",
        font=dict(color="white"),
        legend=dict(x=0.01, y=0.99, bgcolor="rgba(0,0,0,0.4)"),
        margin=dict(l=0, r=0, t=40, b=0),
        height=800,
    )

    return fig


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    graph      = load()
    total      = len(graph["nodes"])
    print(f"Graph has {total} nodes.")

    move_lines = build_move_lines(graph)
    pos3d      = compute_positions_3d(graph, move_lines, max_depth=MAX_DEPTH)
    print(f"Laying out {len(pos3d)} nodes (depth <= {MAX_DEPTH}) ...")

    fig = make_figure(graph, move_lines, pos3d)
    os.makedirs(os.path.dirname(OUTPUT_HTML), exist_ok=True)
    fig.write_html(OUTPUT_HTML, include_plotlyjs="cdn")
    print(f"Saved -> {OUTPUT_HTML}")
    webbrowser.open(os.path.abspath(OUTPUT_HTML))


if __name__ == "__main__":
    main()
