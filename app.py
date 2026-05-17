"""
NAIP Embeddings-Based Similarity Search
========================================
Open-source alternative to Commercial Embeddings-Based Analysis GeoAI tools

Pipeline
--------
Planetary Computer STAC → NAIP COG → chip → ResNet-50 embed → FAISS → results

Run
---
    streamlit run app.py
"""

import io
import logging
import sys
import traceback
from pathlib import Path

# Ensure repo root is on sys.path regardless of where Streamlit Cloud mounts it
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

import config
from utils.imagery import (
    search_naip_scenes,
    load_naip_scene,
    chip_scene,
    build_chip_geodataframe,
    cache_path,
    save_chips,
    load_chips,
    save_meta,
    load_meta,
)
from utils.embeddings import (
    load_model,
    embed_chips,
    build_index,
    query_index,
    save_index,
    load_index,
    save_embeddings,
    load_embeddings,
    umap_project,
)
from utils.viz import build_result_map, build_umap_scatter

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("naip_sim")

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title=config.APP_TITLE,
    page_icon=config.APP_ICON,
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": "https://github.com/rmkenv/naip-similarity",
        "Report a bug": "https://github.com/rmkenv/naip-similarity/issues",
        "About": "NAIP Embeddings Similarity Search — open-source GeoAI",
    },
)

# ── Styles ────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');

  html, body, [class*="css"]          { font-family: 'DM Sans', sans-serif; }
  h1, h2, h3, .mono                  { font-family: 'Space Mono', monospace; }
  .stApp                             { background: #0d1117; color: #c9d1d9; }
  div[data-testid="stSidebar"]       { background: #161b22; border-right: 1px solid #21262d; }
  div[data-testid="stSidebar"] h3    { color: #58a6ff; }

  .badge {
    display: inline-block;
    background: #1f6feb22; color: #58a6ff;
    border: 1px solid #1f6feb55;
    border-radius: 4px; padding: 2px 10px;
    font-size: 0.72rem; font-family: 'Space Mono', monospace;
    letter-spacing: 0.08em; margin-right: 4px;
  }
  .section-header {
    font-family: 'Space Mono', monospace;
    font-size: 0.8rem; letter-spacing: 0.15em;
    color: #58a6ff; text-transform: uppercase;
    border-bottom: 1px solid #21262d;
    padding-bottom: 4px; margin: 1.5rem 0 0.75rem 0;
  }
  .chip-caption {
    font-size: 0.7rem; color: #8b949e;
    font-family: 'Space Mono', monospace; text-align: center;
    margin-top: -8px;
  }
  .metric-card {
    background: #161b22; border: 1px solid #21262d;
    border-radius: 8px; padding: 0.75rem 1rem; margin-bottom: 0.5rem;
  }
  .metric-card .val { font-size: 1.4rem; font-weight: 700; color: #e6edf3; }
  .metric-card .lbl { font-size: 0.72rem; color: #8b949e; }

  .stButton > button {
    background: #21262d; color: #c9d1d9;
    border: 1px solid #30363d; border-radius: 6px;
    font-family: 'Space Mono', monospace; font-size: 0.8rem;
    transition: all 0.15s ease;
  }
  .stButton > button:hover {
    background: #238636; color: #ffffff; border-color: #238636;
  }
  div[data-testid="stExpander"] {
    background: #161b22; border: 1px solid #21262d; border-radius: 8px;
  }
  .stDataFrame { font-size: 0.82rem; }
  .stAlert { border-radius: 8px; }
</style>
""", unsafe_allow_html=True)


# ── Session state init ────────────────────────────────────────────────────────
_STATE_KEYS = [
    "scenes", "selected_scene", "ds", "chips", "positions",
    "embeddings", "faiss_index", "chip_gdf",
    "query_idx", "results", "umap_proj",
    "bbox", "scene_id",
]
for k in _STATE_KEYS:
    if k not in st.session_state:
        st.session_state[k] = None


# ── Cached model load ─────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def _load_model():
    return load_model()


# ── Header ────────────────────────────────────────────────────────────────────
col_title, col_desc = st.columns([3, 2])
with col_title:
    st.markdown("# 🛰️ NAIP · Similarity Search")
    st.markdown(
        '<span class="badge">PLANETARY COMPUTER</span>'
        '<span class="badge">RESNET-50</span>'
        '<span class="badge">FAISS COSINE</span>'
        '<span class="badge">OPEN SOURCE</span>',
        unsafe_allow_html=True,
    )
with col_desc:
    st.markdown("")
    st.caption(
        "Open-source alternative to Commerical Embeddings-Based Analysis.  \n"
        "Transform NAIP imagery into AI embeddings, then find visually similar locations."
    )
st.markdown("---")


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    st.markdown('<p class="section-header">Area of Interest</p>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        west  = st.number_input("West",  value=config.DEFAULT_BBOX["west"],  format="%.4f", step=0.01)
        south = st.number_input("South", value=config.DEFAULT_BBOX["south"], format="%.4f", step=0.01)
    with c2:
        east  = st.number_input("East",  value=config.DEFAULT_BBOX["east"],  format="%.4f", step=0.01)
        north = st.number_input("North", value=config.DEFAULT_BBOX["north"], format="%.4f", step=0.01)

    bbox      = [west, south, east, north]
    bbox_area = (east - west) * (north - south)

    if bbox_area <= 0:
        st.error("Invalid bounding box.")
    elif bbox_area > 0.5:
        st.warning("⚠️ Large AOI — may generate many chips.")

    st.markdown('<p class="section-header">Imagery</p>', unsafe_allow_html=True)
    year = st.selectbox("NAIP Year", config.NAIP_YEARS, index=2)

    st.markdown('<p class="section-header">Chipping</p>', unsafe_allow_html=True)
    chip_size   = st.select_slider("Chip Size (px)", [112, 224, 336], value=224)
    stride_frac = st.slider("Stride (fraction of chip)", 0.25, 1.0, 0.5, 0.25)
    stride      = int(chip_size * stride_frac)

    st.markdown('<p class="section-header">Search</p>', unsafe_allow_html=True)
    top_k     = st.slider("Top-K results", 3, 20, config.DEFAULT_TOP_K)
    show_umap = st.checkbox("Show UMAP embedding space", value=True)

    st.markdown("---")
    search_btn = st.button("🔍 Search NAIP Scenes", use_container_width=True)
    build_btn  = st.button(
        "⚙️ Build Index",
        use_container_width=True,
        disabled=(st.session_state.scenes is None),
    )
    st.markdown("---")
    st.caption(
        "Data: USDA NAIP via Microsoft Planetary Computer  \n"
        "Model: ResNet-50 (ImageNet pretrained)  \n"
        "Index: FAISS IndexFlatIP (cosine)"
    )


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — Search scenes
# ══════════════════════════════════════════════════════════════════════════════
if search_btn:
    if bbox_area <= 0:
        st.error("Fix the bounding box before searching.")
        st.stop()

    with st.spinner("📡 Querying Planetary Computer STAC…"):
        try:
            scenes = search_naip_scenes(bbox, year)
        except Exception as e:
            st.error(f"STAC search failed: {e}")
            log.error(traceback.format_exc())
            st.stop()

    if not scenes:
        st.error(
            f"No NAIP scenes found for this AOI in {year}. "
            "Try a different year or larger extent."
        )
        st.stop()

    st.session_state.scenes = scenes
    st.session_state.bbox   = bbox
    # Reset downstream state when AOI/year changes
    for k in ["ds", "chips", "positions", "embeddings", "faiss_index",
              "chip_gdf", "query_idx", "results", "umap_proj", "scene_id"]:
        st.session_state[k] = None

    st.success(f"Found **{len(scenes)}** scene(s). Select one below and click **Build Index**.")


# ── Scene picker ──────────────────────────────────────────────────────────────
if st.session_state.scenes:
    st.markdown('<p class="section-header">01 · Select Scene</p>', unsafe_allow_html=True)

    scene_labels = [
        f"{i+1}. {s.id}  |  {s.datetime.date() if s.datetime else 'n/a'}"
        for i, s in enumerate(st.session_state.scenes)
    ]
    selected_label = st.radio("Available scenes", scene_labels, label_visibility="collapsed")
    selected_idx   = scene_labels.index(selected_label)
    st.session_state.selected_scene = st.session_state.scenes[selected_idx]

    item = st.session_state.selected_scene
    with st.expander("Scene metadata"):
        meta_df = pd.DataFrame({
            "Field": ["ID", "Date", "CRS", "Cloud Cover", "State"],
            "Value": [
                item.id,
                str(item.datetime.date()) if item.datetime else "n/a",
                item.properties.get("proj:epsg", "n/a"),
                item.properties.get("eo:cloud_cover", "n/a"),
                item.properties.get("naip:state", "n/a"),
            ],
        })
        st.dataframe(meta_df, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — Build index
# ══════════════════════════════════════════════════════════════════════════════
if build_btn and st.session_state.selected_scene is not None:
    item     = st.session_state.selected_scene
    scene_id = item.id

    # Check disk cache
    chips_path = cache_path(scene_id, chip_size, stride, "_chips.npy")
    embs_path  = cache_path(scene_id, chip_size, stride, "_embeddings.npy")
    idx_path   = cache_path(scene_id, chip_size, stride, "_faiss.index")
    meta_path  = cache_path(scene_id, chip_size, stride, "_meta.pkl")
    cache_hit  = all(p.exists() for p in [chips_path, embs_path, idx_path, meta_path])

    if cache_hit:
        st.info("💾 Loading from disk cache…")
        chips     = load_chips(chips_path)
        embs      = load_embeddings(embs_path)
        faiss_idx = load_index(idx_path)
        cached    = load_meta(meta_path)
        positions = cached["positions"]
        chip_gdf  = cached["chip_gdf"]
        log.info(f"Cache hit: {scene_id} | {len(chips)} chips")

    else:
        # Load imagery
        with st.spinner("📥 Loading NAIP COG…"):
            try:
                ds, ov_level = load_naip_scene(item)
            except Exception as e:
                st.error(f"Failed to load imagery: {e}")
                log.error(traceback.format_exc())
                st.stop()

        st.success(
            f"Loaded `{scene_id}` — {ds.shape[1]}×{ds.shape[2]} px "
            f"(overview level {ov_level})"
        )

        # Chip
        with st.spinner("✂️ Chipping imagery…"):
            chips, positions = chip_scene(ds, chip_size=chip_size, stride=stride)

        n_chips = len(chips)
        if n_chips == 0:
            st.error(
                "No chips generated — AOI may be smaller than one chip. "
                "Reduce chip size or enlarge AOI."
            )
            st.stop()
        if n_chips >= config.MAX_CHIPS:
            st.warning(
                f"⚠️ Chip count capped at {config.MAX_CHIPS}. "
                "Increase stride or reduce AOI for a complete index."
            )

        st.info(f"Generated **{n_chips}** chips  ({chip_size}px, stride {stride}px)")

        # Build chip GeoDataFrame
        with st.spinner("📐 Building chip GeoDataFrame…"):
            chip_gdf = build_chip_geodataframe(positions, ds, chip_size)

        # Embed
        model, device = _load_model()
        progress_bar  = st.progress(0.0, text="🧠 Embedding chips…")

        def _cb(done, total):
            progress_bar.progress(done / total,
                                  text=f"🧠 Embedding {done}/{total} chips…")

        try:
            embs = embed_chips(chips, model, device, progress_callback=_cb)
        except Exception as e:
            st.error(f"Embedding failed: {e}")
            log.error(traceback.format_exc())
            st.stop()
        finally:
            progress_bar.empty()

        # FAISS
        with st.spinner("🗂️ Building FAISS index…"):
            faiss_idx = build_index(embs)

        # Persist
        with st.spinner("💾 Saving to cache…"):
            save_chips(chips, chips_path)
            save_embeddings(embs, embs_path)
            save_index(faiss_idx, idx_path)
            save_meta({"positions": positions, "chip_gdf": chip_gdf}, meta_path)

    # Store in session
    st.session_state.chips       = chips
    st.session_state.positions   = positions
    st.session_state.embeddings  = embs
    st.session_state.faiss_index = faiss_idx
    st.session_state.chip_gdf    = chip_gdf
    st.session_state.scene_id    = scene_id
    st.session_state.query_idx   = None
    st.session_state.results     = None
    st.session_state.umap_proj   = None

    # Summary metrics
    mc = st.columns(4)
    mc[0].markdown(f'<div class="metric-card"><div class="val">{len(chips):,}</div><div class="lbl">Chips</div></div>', unsafe_allow_html=True)
    mc[1].markdown(f'<div class="metric-card"><div class="val">{embs.shape[1]:,}</div><div class="lbl">Embed dim</div></div>', unsafe_allow_html=True)
    mc[2].markdown(f'<div class="metric-card"><div class="val">{chip_size}px</div><div class="lbl">Chip size</div></div>', unsafe_allow_html=True)
    mc[3].markdown(f'<div class="metric-card"><div class="val">{"cache ✓" if cache_hit else "fresh"}</div><div class="lbl">Source</div></div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — Query
# ══════════════════════════════════════════════════════════════════════════════
if st.session_state.chips is not None:
    chips     = st.session_state.chips
    n_chips   = len(chips)
    embs      = st.session_state.embeddings
    faiss_idx = st.session_state.faiss_index
    chip_gdf  = st.session_state.chip_gdf

    st.markdown("---")
    st.markdown('<p class="section-header">02 · Select Query Chip</p>', unsafe_allow_html=True)

    qc1, qc2, qc3 = st.columns([2, 1, 3])
    with qc1:
        current_q = st.session_state.query_idx if st.session_state.query_idx is not None else 0
        query_idx = st.number_input(
            f"Chip index  (0 – {n_chips - 1})",
            min_value=0, max_value=n_chips - 1,
            value=current_q, step=1,
        )
    with qc2:
        st.markdown("&nbsp;", unsafe_allow_html=True)
        if st.button("🎲 Random"):
            st.session_state.query_idx = int(np.random.randint(0, n_chips))
            st.rerun()

    with qc3:
        chip_rgb = (chips[query_idx].transpose(1, 2, 0) * 255).astype(np.uint8)
        st.image(chip_rgb, caption=f"Query — chip #{query_idx}", width=180)

    # Chip browser
    with st.expander("🖼️ Browse chips (sample)"):
        sample_n   = min(24, n_chips)
        step_size  = max(1, n_chips // sample_n)
        sample_ids = list(range(0, n_chips, step_size))[:sample_n]
        gcols      = st.columns(8)
        for j, sid in enumerate(sample_ids):
            rgb = (chips[sid].transpose(1, 2, 0) * 255).astype(np.uint8)
            gcols[j % 8].image(rgb, use_container_width=True)
            gcols[j % 8].markdown(
                f'<p class="chip-caption">#{sid}</p>', unsafe_allow_html=True
            )

    find_btn = st.button(
        f"🔎 Find top-{top_k} similar chips",
        type="primary",
    )

    if find_btn:
        result_indices, result_scores = query_index(faiss_idx, embs, query_idx, top_k)
        st.session_state.query_idx = query_idx
        st.session_state.results   = (result_indices, result_scores)

        if show_umap and st.session_state.umap_proj is None:
            with st.spinner("📐 Computing UMAP projection… (~30 s one-time)"):
                try:
                    st.session_state.umap_proj = umap_project(embs)
                except ImportError:
                    st.warning("Install `umap-learn` for UMAP visualization.")
                except Exception as e:
                    st.warning(f"UMAP failed: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4 — Results
# ══════════════════════════════════════════════════════════════════════════════
if st.session_state.results is not None:
    result_indices, result_scores = st.session_state.results
    query_idx = st.session_state.query_idx
    chip_gdf  = st.session_state.chip_gdf
    chips     = st.session_state.chips
    bbox_used = st.session_state.bbox

    st.markdown("---")
    st.markdown(
        f'<p class="section-header">03 · Top-{len(result_indices)} Similar Chips</p>',
        unsafe_allow_html=True,
    )

    # Chip grid
    n_cols   = min(len(result_indices), 8)
    res_cols = st.columns(n_cols)
    for col, (idx, score) in zip(res_cols, zip(result_indices, result_scores)):
        rgb = (chips[idx].transpose(1, 2, 0) * 255).astype(np.uint8)
        col.image(rgb, use_container_width=True)
        col.markdown(
            f'<p class="chip-caption">#{idx}<br>{score:.4f}</p>',
            unsafe_allow_html=True,
        )

    tab_map, tab_umap, tab_table, tab_export = st.tabs(
        ["🗺️ Map", "📐 Embedding Space", "📊 Table", "💾 Export"]
    )

    with tab_map:
        if bbox_used:
            center = ((bbox_used[1] + bbox_used[3]) / 2,
                      (bbox_used[0] + bbox_used[2]) / 2)
        else:
            c = chip_gdf.dissolve().centroid.iloc[0]
            center = (c.y, c.x)

        with st.spinner("Rendering map…"):
            fmap = build_result_map(
                query_idx, result_indices, result_scores, chip_gdf, center
            )
        st_folium(fmap, width="100%", height=520, returned_objects=[])

    with tab_umap:
        if st.session_state.umap_proj is not None:
            fig = build_umap_scatter(
                st.session_state.umap_proj,
                query_idx, result_indices, result_scores,
                len(chips),
            )
            st.plotly_chart(fig, use_container_width=True)
        elif show_umap:
            st.info("UMAP not yet computed. Click **Find similar chips** to trigger it.")
        else:
            st.info("Enable **Show UMAP embedding space** in the sidebar, then run a search.")

    with tab_table:
        df = pd.DataFrame({
            "Rank":       range(1, len(result_indices) + 1),
            "Chip ID":    result_indices,
            "Cosine Sim": [f"{s:.6f}" for s in result_scores],
            "Pixel Row":  [chip_gdf.loc[chip_gdf["chip_id"] == i, "pixel_row"].iloc[0]
                           for i in result_indices],
            "Pixel Col":  [chip_gdf.loc[chip_gdf["chip_id"] == i, "pixel_col"].iloc[0]
                           for i in result_indices],
        })
        st.dataframe(df.set_index("Rank"), use_container_width=True)

    with tab_export:
        st.markdown("#### Download results")

        result_gdf = chip_gdf[chip_gdf["chip_id"].isin(result_indices)].copy()
        score_map  = dict(zip(result_indices, result_scores))
        result_gdf["cosine_sim"] = result_gdf["chip_id"].map(score_map)
        result_gdf["query_chip"] = query_idx
        result_gdf["scene_id"]   = st.session_state.scene_id

        st.download_button(
            "⬇️ GeoJSON — similar chip polygons",
            data=result_gdf.to_json().encode(),
            file_name=f"naip_similar_chips_{st.session_state.scene_id}.geojson",
            mime="application/geo+json",
            use_container_width=True,
        )

        df_export = pd.DataFrame({
            "Rank":       range(1, len(result_indices) + 1),
            "Chip ID":    result_indices,
            "Cosine Sim": result_scores,
        })
        st.download_button(
            "⬇️ CSV — similarity scores",
            data=df_export.to_csv(index=False).encode(),
            file_name=f"naip_scores_{st.session_state.scene_id}.csv",
            mime="text/csv",
            use_container_width=True,
        )

        buf = io.BytesIO()
        np.save(buf, st.session_state.embeddings[[query_idx] + result_indices])
        st.download_button(
            "⬇️ NPY — embeddings (query + results)",
            data=buf.getvalue(),
            file_name=f"naip_embeddings_subset_{st.session_state.scene_id}.npy",
            mime="application/octet-stream",
            use_container_width=True,
        )

        st.caption(
            "GeoJSON loads directly into QGIS, ArcGIS Pro, or GeoPandas.  \n"
            "NPY embeddings are L2-normalized float32 arrays, shape (K+1, 2048)."
        )
