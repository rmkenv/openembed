# NAIP Similarity Search — dev & deploy helpers (no Docker)

.PHONY: install run clean cache-clear lint freeze

# ── Local dev ─────────────────────────────────────────────────────────────────
install:
	python -m pip install --upgrade pip
	pip install -r requirements.txt

run:
	streamlit run app.py

# ── Cache management ──────────────────────────────────────────────────────────
cache-clear:
	rm -rf cache/*.npy cache/*.index cache/*.pkl
	@echo "Cache cleared."

# ── Code quality ──────────────────────────────────────────────────────────────
lint:
	pip install --quiet ruff
	ruff check app.py config.py utils/

# ── Dependency management ─────────────────────────────────────────────────────
freeze:
	pip freeze > requirements.lock
	@echo "Pinned requirements written to requirements.lock"
