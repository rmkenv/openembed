"""
utils/viz.py — Folium map builders and Plotly UMAP scatter
"""
import numpy as np
import folium
import plotly.graph_objects as go
from shapely.geometry import mapping

from pathlib import Path
import sys
_ROOT = str(Path(__file__).parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
from config import ESRI_TILES, ESRI_ATTR


# ── Color helpers ─────────────────────────────────────────────────────────────

def score_to_color(score: float, min_s: float, max_s: float) -> str:
    """Map cosine similarity score to a green→yellow hex color."""
    norm = (score - min_s) / (max_s - min_s + 1e-8)
    r = int(255 * (1 - norm))
    g = 210
    b = 60
    return f"#{r:02x}{g:02x}{b:02x}"


# ── Folium result map ─────────────────────────────────────────────────────────

def build_result_map(
    query_idx: int,
    result_indices: list[int],
    result_scores: list[float],
    chip_gdf,               # GeoDataFrame with geometry column, indexed by chip_id
    center: tuple[float, float],
    zoom: int = 14,
) -> folium.Map:
    """
    Build a folium map showing:
      - Cyan polygon  : query chip
      - Scored polygons: result chips colored by cosine similarity
    """
    m = folium.Map(
        location=list(center),
        zoom_start=zoom,
        tiles=ESRI_TILES,
        attr=ESRI_ATTR,
    )

    # Layer control + minimap
    folium.LayerControl().add_to(m)

    # Query chip
    q_geom = chip_gdf.loc[chip_gdf["chip_id"] == query_idx, "geometry"].iloc[0]
    folium.GeoJson(
        mapping(q_geom),
        name="Query chip",
        style_function=lambda _: {
            "fillColor": "#00e5ff",
            "color": "#00e5ff",
            "weight": 2.5,
            "fillOpacity": 0.45,
        },
        tooltip=folium.Tooltip(f"<b>QUERY</b> — chip #{query_idx}"),
    ).add_to(m)

    # Result chips
    min_s = min(result_scores)
    max_s = max(result_scores)
    for rank, (idx, score) in enumerate(zip(result_indices, result_scores), 1):
        geom = chip_gdf.loc[chip_gdf["chip_id"] == idx, "geometry"].iloc[0]
        color = score_to_color(score, min_s, max_s)
        folium.GeoJson(
            mapping(geom),
            name=f"Result #{rank}",
            style_function=lambda _, c=color: {
                "fillColor": c,
                "color": "#ffffff",
                "weight": 1.5,
                "fillOpacity": 0.55,
            },
            tooltip=folium.Tooltip(
                f"<b>Rank {rank}</b><br>Chip #{idx}<br>Cosine sim: {score:.4f}"
            ),
        ).add_to(m)

    return m


# ── UMAP scatter plot ──────────────────────────────────────────────────────────

def build_umap_scatter(
    proj: np.ndarray,
    query_idx: int,
    result_indices: list[int],
    result_scores: list[float],
    n_chips: int,
) -> go.Figure:
    """
    Plotly scatter of 2-D UMAP projection.
    Background chips: grey | Result chips: colored by score | Query: cyan star
    """
    result_set = set(result_indices)

    # Background
    bg_mask = [i for i in range(n_chips) if i not in result_set and i != query_idx]
    bg_x = proj[bg_mask, 0]
    bg_y = proj[bg_mask, 1]

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=bg_x, y=bg_y,
        mode="markers",
        marker=dict(size=4, color="#3a3f47", opacity=0.5),
        name="All chips",
        hovertemplate="chip %{text}<extra></extra>",
        text=[str(i) for i in bg_mask],
    ))

    # Results colored by score
    min_s = min(result_scores) if result_scores else 0
    max_s = max(result_scores) if result_scores else 1
    norm_scores = [(s - min_s) / (max_s - min_s + 1e-8) for s in result_scores]

    fig.add_trace(go.Scatter(
        x=proj[result_indices, 0],
        y=proj[result_indices, 1],
        mode="markers",
        marker=dict(
            size=10,
            color=norm_scores,
            colorscale="YlGn",
            colorbar=dict(title="Similarity", thickness=12),
            line=dict(width=1, color="white"),
        ),
        name="Similar chips",
        hovertemplate="chip %{text}<br>sim: %{customdata:.4f}<extra></extra>",
        text=[str(i) for i in result_indices],
        customdata=result_scores,
    ))

    # Query chip
    fig.add_trace(go.Scatter(
        x=[proj[query_idx, 0]],
        y=[proj[query_idx, 1]],
        mode="markers",
        marker=dict(size=16, symbol="star", color="#00e5ff",
                    line=dict(width=1.5, color="white")),
        name="Query",
        hovertemplate=f"QUERY — chip {query_idx}<extra></extra>",
    ))

    fig.update_layout(
        paper_bgcolor="#0d1117",
        plot_bgcolor="#161b22",
        font=dict(color="#e6edf3", family="Space Mono, monospace"),
        legend=dict(bgcolor="#161b22", bordercolor="#30363d", borderwidth=1),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        margin=dict(l=20, r=20, t=30, b=20),
        title=dict(text="Embedding Space (UMAP)", font=dict(size=13)),
        height=420,
    )
    return fig
