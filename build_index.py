

import os
import glob
import json
import time
import pickle

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
TEXTBOOK_DIR = os.path.join(HERE, "data_clean", "textbooks", "en")
STORE_DIR = os.path.join(HERE, "rag_store")
EMBED_MODEL = "BAAI/bge-small-en-v1.5"   
WORDS_PER_CHUNK = 220                    
WORD_OVERLAP = 40                         
BATCH_SIZE = 64                           
MAX_SEQ_LEN = 192                        
DEVICE = "mps"                            


def pick_device():
    try:
        import torch
        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


def chunk_text(text, words_per_chunk=WORDS_PER_CHUNK, overlap=WORD_OVERLAP):
    """Split a long document into overlapping word windows."""
    words = text.split()
    chunks = []
    step = words_per_chunk - overlap
    for start in range(0, len(words), step):
        window = words[start:start + words_per_chunk]
        if len(window) < 25:          # drop tiny tail fragments
            continue
        chunks.append(" ".join(window))
    return chunks


def main():
    device = pick_device()
    os.makedirs(STORE_DIR, exist_ok=True)
    files = sorted(glob.glob(os.path.join(TEXTBOOK_DIR, "*.txt")))
    print(f"Found {len(files)} textbooks. Embedding device = {device}")

    all_chunks = []          # the passage text
    all_sources = []         # which textbook it came from
    for fp in files:
        book = os.path.splitext(os.path.basename(fp))[0]
        with open(fp, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
        chunks = chunk_text(text)
        all_chunks.extend(chunks)
        all_sources.extend([book] * len(chunks))
        print(f"  {book:28s} -> {len(chunks):6d} chunks")

    print(f"\nTotal chunks: {len(all_chunks):,}")

    print(f"Loading embedding model {EMBED_MODEL} ...")
    model = SentenceTransformer(EMBED_MODEL, device=device)
    model.max_seq_length = MAX_SEQ_LEN

    t0 = time.time()
    try:
        emb = model.encode(
            all_chunks,
            batch_size=BATCH_SIZE,
            show_progress_bar=True,
            normalize_embeddings=True,     # so inner-product == cosine similarity
            convert_to_numpy=True,
        ).astype("float32")
    except RuntimeError as e:
        print(f"\n{device} failed ({e}); retrying on CPU ...")
        model = model.to("cpu")
        emb = model.encode(
            all_chunks,
            batch_size=64,
            show_progress_bar=True,
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).astype("float32")
    print(f"Embedded {len(all_chunks):,} chunks in {time.time()-t0:.1f}s "
          f"-> shape {emb.shape}")

    dim = emb.shape[1]
    index = faiss.IndexFlatIP(dim)     # cosine sim via normalized inner product
    index.add(emb)

    faiss.write_index(index, os.path.join(STORE_DIR, "faiss.index"))
    with open(os.path.join(STORE_DIR, "chunks.pkl"), "wb") as f:
        pickle.dump({"chunks": all_chunks, "sources": all_sources}, f)
    meta = {
        "embed_model": EMBED_MODEL,
        "dim": int(dim),
        "n_chunks": len(all_chunks),
        "words_per_chunk": WORDS_PER_CHUNK,
        "overlap": WORD_OVERLAP,
        "n_textbooks": len(files),
        "textbooks": [os.path.basename(f) for f in files],
    }
    with open(os.path.join(STORE_DIR, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nSaved FAISS store to {STORE_DIR}/")
    print(json.dumps(meta, indent=2)[:400])


if __name__ == "__main__":
    main()
