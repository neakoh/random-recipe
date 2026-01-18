# Random Dish Picker

Pick a random recipe from your EPUB cookbook library.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
streamlit run app.py
```

## Import via CLI (optional)

```bash
python importer.py /path/to/cookbooks --db data/library.sqlite
```

## Notes

- Handles both `.epub` files and `.epub` directories (unpacked EPUBs).
- Stores recipe candidates in SQLite with best-effort heuristics.
- EPUB locations are stored as `file.xhtml#anchor` when available.
