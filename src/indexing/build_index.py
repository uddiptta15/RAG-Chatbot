"""
FAISS index build/persist. -> component docs/indexing.md, section 4

Entry point: run `python -m src.indexing.build_index` to (re)build `data/faiss.index` and
`data/metadata_store.json` from `data/d2l-en`. `load_index`/`load_metadata_store`/
`load_metadata_by_chunk_id` are the read side, shared by src/retrieval/*.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import faiss
import numpy as np

from src.indexing.chunker import chunk_repo
from src.indexing.embedder import embed_chunks

DEFAULT_FAISS_PATH = Path("data/faiss.index")
DEFAULT_METADATA_PATH = Path("data/metadata_store.json")


def build_index(embeddings: np.ndarray) -> faiss.Index:
    """`IndexFlatIP` over normalized embeddings, i.e. cosine similarity. -> section 4."""
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    return index


def save(
    index: faiss.Index,
    metadata_store: dict,
    faiss_path: Path = DEFAULT_FAISS_PATH,
    metadata_path: Path = DEFAULT_METADATA_PATH,
) -> None:
    faiss_path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(faiss_path))
    metadata_path.write_text(json.dumps(metadata_store, indent=2), encoding="utf-8")


EMBED_BATCH_SIZE = 200  # a checkpoint interval, not an API-quota limit -- embedding is
                        # local now (src/indexing/embedder.py's fastembed backend), but
                        # incremental saves are still cheap insurance against any
                        # mid-run crash (power loss, disk full, etc.) on a 2660+ chunk job


def build_and_save(
    root_dir: str | Path = "data/d2l-en",
    faiss_path: Path = DEFAULT_FAISS_PATH,
    metadata_path: Path = DEFAULT_METADATA_PATH,
    batch_size: int = EMBED_BATCH_SIZE,
) -> tuple[int, int]:
    """Full pipeline: `chunk_repo` -> `embed_chunks` -> `build_index` -> `save`. The
    metadata store is keyed by FAISS internal index position (insertion order, as a
    string) rather than `chunk_id` -- section 4: "keyed by FAISS internal index position
    or chunk_id" -- so `retrieval/dense.py` can map a search hit straight back to a chunk
    without a second lookup table.

    **Resumable and checkpointed** -- originally added because the free *Gemini* embedding
    tier's daily quota (~1000/day) couldn't cover this corpus (2660+ chunks) in one run.
    Embeddings moved to a local model since (indexing.md, section 3), which removed that
    specific constraint, but the resumability is kept regardless: it's cheap insurance
    against any other mid-run failure, and this saves to disk after every batch. If
    `faiss_path`/`metadata_path` already exist from a prior (possibly partial) run, this
    picks up embedding from the first chunk not yet indexed rather than starting over --
    safe to call repeatedly. Returns `(total_chunk_count, embedding_dim)` --
    `embedding_dim` is 0 if nothing has been indexed yet (e.g. the very first batch itself
    failed).

    Caution: resuming assumes `chunk_repo(root_dir)` produces the same chunks in the same
    order as the prior run (true as long as the corpus files and chunker.py haven't
    changed). If either changed, delete `faiss_path`/`metadata_path` first to force a clean
    rebuild instead of a misaligned resume."""
    chunks = chunk_repo(root_dir)

    faiss_path, metadata_path = Path(faiss_path), Path(metadata_path)
    if faiss_path.exists() and metadata_path.exists():
        index = faiss.read_index(str(faiss_path))
        metadata_store = json.loads(metadata_path.read_text(encoding="utf-8"))
        already_done = index.ntotal
        print(f"resuming: {already_done}/{len(chunks)} chunks already indexed")
    else:
        index = None
        metadata_store = {}
        already_done = 0

    remaining = chunks[already_done:]
    for batch_start in range(0, len(remaining), batch_size):
        batch = remaining[batch_start : batch_start + batch_size]
        embeddings = embed_chunks(batch)  # may raise -- everything up to the PREVIOUS
                                           # batch is already saved by then, so nothing is lost

        if index is None:
            index = build_index(embeddings)
        else:
            index.add(embeddings)

        offset = already_done + batch_start
        for i, chunk in enumerate(batch):
            metadata_store[str(offset + i)] = chunk.to_dict()
        save(index, metadata_store, faiss_path, metadata_path)
        print(f"  indexed {offset + len(batch)}/{len(chunks)} chunks")

    return len(chunks), (index.d if index is not None else 0)


@lru_cache(maxsize=1)
def load_index(faiss_path: Path = DEFAULT_FAISS_PATH) -> faiss.Index:
    if not Path(faiss_path).exists():
        raise FileNotFoundError(f"{faiss_path} not found -- run `python -m src.indexing.build_index` first")
    return faiss.read_index(str(faiss_path))


@lru_cache(maxsize=1)
def load_metadata_store(metadata_path: Path = DEFAULT_METADATA_PATH) -> tuple[dict, ...]:
    """Chunks ordered by FAISS position -- `result[i]` is the chunk at FAISS position `i`.
    Returned as a tuple (rather than a list) so this sits behind `lru_cache`."""
    if not Path(metadata_path).exists():
        raise FileNotFoundError(f"{metadata_path} not found -- run `python -m src.indexing.build_index` first")
    raw = json.loads(Path(metadata_path).read_text(encoding="utf-8"))
    return tuple(raw[str(i)] for i in range(len(raw)))


@lru_cache(maxsize=1)
def load_metadata_by_chunk_id(metadata_path: Path = DEFAULT_METADATA_PATH) -> dict:
    """The same chunks as `load_metadata_store`, keyed by `chunk_id` instead of position --
    for hydrating fusion's already-ranked chunk_ids (retrieval.md, section 5)."""
    return {chunk["chunk_id"]: chunk for chunk in load_metadata_store(metadata_path)}


if __name__ == "__main__":
    n_chunks, dim = build_and_save()
    print(f"indexed {n_chunks} chunks, dim={dim}")
    print(f"saved -> {DEFAULT_FAISS_PATH}, {DEFAULT_METADATA_PATH}")
