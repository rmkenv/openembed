# 🛰️ NAIP Embeddings-Based Similarity Search

Open-source alternative to **ArcGIS Pro 3.7 Embeddings-Based Analysis GeoAI toolset**, built on free/open infrastructure. No proprietary software, no Docker, no API keys.

## What it does

Given a NAIP scene from Microsoft Planetary Computer, the app:

1. **Chips** imagery into N×N pixel patches
2. **Embeds** each chip through a ResNet-50 backbone → 2048-d vectors
3. **Indexes** all embeddings in a FAISS cosine similarity index
4. **Queries** — user selects any chip; app returns the top-K visually similar chips
5. **Visualizes** results on a satellite basemap + interactive UMAP embedding scatter
6. **Exports** results as GeoJSON, CSV, or NumPy embeddings

## Stack

| Component | Library |
|-----------|---------|
| Imagery | USDA NAIP via Microsoft Planetary Computer STAC |
| COG reading | `rioxarray` |
| Embeddings | ResNet-50 (ImageNet V2) → 2048-d, `torchvision` |
| Similarity search | `faiss-cpu` IndexFlatIP (cosine) |
| Embedding viz | `umap-learn` + `plotly` |
| Geo output | `geopandas` → GeoJSON / CSV |
| Map | `folium` + `streamlit-folium` (Esri World Imagery) |
| UI | `streamlit` |

## Project structure

```
naip_similarity/
├── app.py                  # Main Streamlit app
├── config.py               # All tuneable constants
├── utils/
│   ├── imagery.py          # STAC search, COG load, chipping, disk cache
│   ├── embeddings.py       # ResNet-50 embed, FAISS build/query, UMAP
│   └── viz.py              # Folium map builder, Plotly UMAP scatter
├── cache/                  # Auto-created: persisted chips/embeddings/index
├── .streamlit/
│   └── config.toml         # Theme + server settings
├── Makefile                # Dev helpers
└── requirements.txt
```

## Quickstart (local)

```bash
git clone https://github.com/rmkenv/naip-similarity
cd naip-similarity

# Recommended: use a virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

make install                     # pip install -r requirements.txt
make run                         # streamlit run app.py
```

Open `http://localhost:8501`.

## Deploy to Streamlit Community Cloud (free)

1. Push the repo to GitHub (public or private)
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app**
3. Select repo, branch `main`, main file `app.py`
4. Click **Deploy** — no secrets or env vars needed

> **Note:** The `cache/` directory is ephemeral on Streamlit Cloud — embeddings re-compute on each cold start. For persistence, point `CACHE_DIR` in `config.py` to a mounted volume or S3-backed path.

## Deploy to a Linux VM / VPS (no Docker)

```bash
# On the server
git clone https://github.com/rmkenv/naip-similarity
cd naip-similarity
python -m venv .venv && source .venv/bin/activate
make install

# Run persistently with screen or tmux
screen -S naip
make run
# Ctrl+A D to detach

# Or create a systemd service (see below)
```

### Systemd service (optional)

Create `/etc/systemd/system/naip-sim.service`:

```ini
[Unit]
Description=NAIP Similarity Search
After=network.target

[Service]
User=ubuntu
WorkingDirectory=/home/ubuntu/naip-similarity
ExecStart=/home/ubuntu/naip-similarity/.venv/bin/streamlit run app.py \
    --server.port=8501 --server.address=0.0.0.0 --server.headless=true
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable naip-sim
sudo systemctl start naip-sim
```

## Disk cache

Chips, embeddings, and FAISS index are persisted to `cache/` keyed by  
`md5(scene_id + chip_size + stride)`. Subsequent runs with the same parameters load instantly.

```bash
make cache-clear    # wipe cache
make freeze         # pin exact versions to requirements.lock
make lint           # ruff check
```

## Upgrade paths

| Upgrade | How |
|---------|-----|
| Geospatial embeddings | Swap ResNet-50 for [Clay Foundation Model](https://github.com/Clay-foundation/model) — pretrained on NAIP/Sentinel/Landsat |
| GPU | `faiss-gpu` + CUDA PyTorch build; set `device="cuda"` in `config.py` |
| Persistent cache across restarts | Mount an external volume at `cache/` or push to S3 with `s3fs` |
| Draw AOI on map | `streamlit-folium` Draw plugin instead of bbox inputs |
| RGBI (4-band) embeddings | Keep band 4 in `chip_scene()`; adjust ResNet first conv layer |
| Approximate search at scale (>100k chips) | Swap `IndexFlatIP` for `IndexIVFFlat` or `IndexHNSWFlat` |

## License

MIT
