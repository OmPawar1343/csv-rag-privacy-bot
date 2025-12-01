# populate.py
"""
# Purpose: Index CSVs into ChromaDB for RAG — read rows in chunks, 
# turn each row into "Column: Value" text, embed in batches, 
# and upsert docs+vectors+metadata to the persistent collection used later by rag.py.
"""

import os, glob, hashlib
import pandas as pd
from chromadb import PersistentClient
from embeddings.embedding_model import get_embedding_model

# Config (override via env if needed)
# Why: Make paths and performance knobs configurable without code changes.
DB_PATH = os.getenv("CHROMA_DB_PATH", "db/chroma_db")
COLLECTION_NAME = os.getenv("CHROMA_COLLECTION", "csv_collection")
DATA_DIR = os.getenv("DATA_DIR", "data")
EMBED_BATCH_SIZE = int(os.getenv("INDEX_BATCH_SIZE", "256"))      # embeddings per upsert
READ_CHUNK_ROWS = int(os.getenv("READ_CHUNK_ROWS", "5000"))       # rows per pandas chunk
WIPE = os.getenv("WIPE_INDEX", "1").lower() in ("1", "true", "yes", "on")

def row_to_doc(row: pd.Series) -> str:
    """
    What: Convert a pandas row to a simple KV text document ("Col: Val" per line).
    Why: Keeps downstream retrieval consistent and schema-free; rag.py can parse KVs.
    """
    parts = []
    for col, val in row.items():
        if pd.isna(val):
            continue
        # Why: Normalize whitespace/newlines so every doc is single-line per KV.
        s = str(val).replace("\r", " ").replace("\n", " ").strip()
        parts.append(f"{col}: {s}")
    return "\n".join(parts) if parts else "(empty row)"

def file_id(path: str) -> str:
    """
    What: Compute a short, deterministic identifier for a CSV file using SHA-1.
    Why: Helps create stable document IDs (file hash + row idx) for idempotent upserts.
    """
    h = hashlib.sha1()
    # Why: Stream the file in 1MB chunks to support large files without high memory usage.
    with open(path, "rb") as f:
        while True:
            b = f.read(1024 * 1024)
            if not b:
                break
            h.update(b)
    base = os.path.basename(path)
    return f"{base}::{h.hexdigest()[:10]}"

def _encode(embedder, texts):
    """
    What: Encode a batch of texts into embeddings, compatible with either:
          - a wrapper returned by get_embedding_model(), or
          - a raw SentenceTransformer model.
    Why: Keep the ingestion code agnostic to the embedding backend.
    """
    # Works whether get_embedding_model() returns a wrapper or a raw SentenceTransformer
    try:
        vecs = embedder.encode(texts)  # wrapper path (returns numpy array)
    except TypeError:
        # raw SentenceTransformer path (avoid tqdm output)
        vecs = embedder.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    # Why: Ensure numpy array type even if backend returns list (some models do).
    try:
        import numpy as np
        if isinstance(vecs, list):
            vecs = np.array(vecs)
    except Exception:
        pass
    return vecs

def index_csv(csv_path: str, collection, embedder):
    """
    What: Stream a CSV and upsert rows into ChromaDB with embeddings.
    Why:
      - Chunked reading avoids loading entire file into memory.
      - Batched embeddings improve throughput and reduce API overhead.
      - Deterministic IDs allow re-runs to update instead of duplicate.

    Returns:
      - Total number of rows indexed from this CSV.
    """
    fid = file_id(csv_path)
    docs, metas, ids = [], [], []
    total = 0
    # Stream rows to handle large files
    for chunk in pd.read_csv(
        csv_path, dtype=str, keep_default_na=False, engine="python",
        on_bad_lines="skip", chunksize=READ_CHUNK_ROWS
    ):
        for _, row in chunk.iterrows():
            doc = row_to_doc(row)
            rid = f"{fid}:row_{total}"  # Why: stable and unique per file+row index
            docs.append(doc)
            metas.append({"source": os.path.basename(csv_path), "row_id": int(total), "file_id": fid})
            ids.append(rid)
            total += 1

            # Why: Upsert in embedding-sized batches to balance memory and speed.
            if len(docs) >= EMBED_BATCH_SIZE:
                vecs = _encode(embedder, docs)
                collection.upsert(ids=ids, documents=docs, metadatas=metas, embeddings=vecs.tolist())
                docs, metas, ids = [], [], []

    # Flush any remainder
    if docs:
        vecs = _encode(embedder, docs)
        collection.upsert(ids=ids, documents=docs, metadatas=metas, embeddings=vecs.tolist())

    return total

def main():
    """
    What: Entry point that (optionally) wipes the collection, indexes all CSVs in DATA_DIR,
          and reports totals.
    Why:
      - WIPE_INDEX=1 (default) ensures a clean rebuild during development.
      - In production, set WIPE_INDEX=0 to append/update without dropping the collection.
    """
    os.makedirs(DB_PATH, exist_ok=True)
    client = PersistentClient(path=DB_PATH)

    # Why: Optional wipe for a clean slate; useful in dev or when schema changes.
    if WIPE:
        try:
            client.delete_collection(name=COLLECTION_NAME)
            print(f"Deleted existing collection '{COLLECTION_NAME}'.")
        except Exception:
            pass

    embedder = get_embedding_model()
    # Create or open the collection and record embedder metadata for traceability
    # Why: Capturing model name/normalize flag helps reproduce/debug query behavior later.
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={
            "embed_model": getattr(embedder, "model_name", "unknown"),
            "embed_normalize": bool(getattr(embedder, "normalize", False)),
        },
    )

    print(f"[i] Using embedder: {getattr(embedder, 'model_name', 'unknown')} "
          f"(normalize={bool(getattr(embedder, 'normalize', False))})")

    # Why: Deterministic ordering of files for predictable indexing runs.
    files = sorted(glob.glob(os.path.join(DATA_DIR, "*.csv")))
    if not files:
        print(f"[!] No CSVs found in {DATA_DIR}")
        return

    grand_total = 0
    for f in files:
        print(f"[i] Indexing {f} ...")
        count = index_csv(f, collection, embedder)
        print(f"    -> {count} rows")
        grand_total += count

    print(f"[✓] Indexed {grand_total} rows into collection '{COLLECTION_NAME}' at '{DB_PATH}'.")

if __name__ == "__main__":
    main()