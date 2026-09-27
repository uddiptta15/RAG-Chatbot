# data/

This folder's contents are gitignored (corpus + generated index are large and/or
regenerable) — after cloning, this directory will be empty except for this file. To
reproduce it:

1. **Clone the corpus** (see `component docs/indexing.md`, section 1):
   ```
   git clone --depth 1 --branch master https://github.com/d2l-ai/d2l-en data/d2l-en
   rm -rf data/d2l-en/.git
   find data/d2l-en -mindepth 1 -maxdepth 1 ! -name "chapter_*" -exec rm -rf {} +
   ```
   (Only the 27 `chapter_*/` folders and their `.md` files are needed — everything else
   in the upstream repo is assets/tooling, not course content.)

2. **Build the index** — chunks the corpus, embeds every chunk locally (no API key or
   quota needed for this step — see `indexing.md`, section 3), and writes `faiss.index` +
   `metadata_store.json`:
   ```
   python -m src.indexing.build_index
   ```
   This is resumable — safe to re-run if it's interrupted partway through; it picks up
   from the last chunk indexed rather than starting over.

Expected end state: `data/d2l-en/` (191 `.md` files across 27 chapters), `data/faiss.index`,
`data/metadata_store.json`.
