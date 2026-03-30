import os
import argparse
import pickle
import time
from pathlib import Path
from collections import defaultdict

import numpy as np
import faiss
from pypdf import PdfReader
import google.generativeai as genai

# ──────────────────────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────────────────────
# Use your free-tier API key here
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "YOUR_GEMINI_API_KEY_HERE")
GEMINI_API_KEY = "AIzaSyAElp5Nslfg3bMFs6nsWJQDuuw76MSAD8A"
genai.configure(api_key=GEMINI_API_KEY)


EMBED_MODEL   = "models/gemini-embedding-2-preview"
CHUNK_SIZE    = 800
CHUNK_OVERLAP = 100

# TARGET ONLY THE ONES THAT FAILED
PAYERS = ["cigna", "aetna"]

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def extract_payer_disease(filename: str):
    stem = Path(filename).stem.lower()
    stem = stem.replace("_policy", "")
    parts = stem.split("_", 1)
    payer = parts[0]
    disease = parts[1] if len(parts) > 1 else "unknown"
    return payer, disease

def read_pdf_text(path: str) -> str:
    try:
        reader = PdfReader(path)
        text = ""
        for page in reader.pages:
            content = page.extract_text()
            if content:
                text += content + "\n"
        return text
    except Exception as e:
        print(f"  [ERROR] Could not read {path}: {e}")
        return ""

def chunk_text(text: str) -> list[str]:
    chunks = []
    if not text: return chunks
    start = 0
    while start < len(text):
        end = start + CHUNK_SIZE
        chunks.append(text[start:end])
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks

def get_embeddings_with_retry(texts: list[str], retries=3):
    """Fetches embeddings with a retry mechanism for 429 errors."""
    for attempt in range(retries):
        try:
            result = genai.embed_content(
                model=EMBED_MODEL,
                content=texts,
                task_type="retrieval_document"
            )
            return result['embedding']
        except Exception as e:
            if "429" in str(e) and attempt < retries - 1:
                wait_time = (attempt + 1) * 10  # Wait 10s, then 20s
                print(f"  ⚠️ Rate limit hit. Sleeping {wait_time}s before retry...")
                time.sleep(wait_time)
            else:
                raise e

def build_and_save_faiss(chunks: list[str], metadata: list[dict], save_dir: Path):
    save_dir.mkdir(parents=True, exist_ok=True)
    print(f"  ⏳ Embedding {len(chunks)} chunks (Rate-Limited Mode)...")
    
    # Smaller batch size for Free Tier
    batch_size = 10 
    all_vectors = []
    
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        vectors = get_embeddings_with_retry(batch)
        all_vectors.extend(vectors)
        
        # Consistent pause between batches (approx 30 requests per minute)
        print(f"     ∟ Progress: {len(all_vectors)}/{len(chunks)}...")
        time.sleep(2.5) 

    matrix = np.array(all_vectors, dtype="float32")
    index = faiss.IndexFlatL2(matrix.shape[1])
    index.add(matrix)

    faiss.write_index(index, str(save_dir / "index.faiss"))
    with open(save_dir / "metadata.pkl", "wb") as f:
        pickle.dump(metadata, f)
    print(f"  ✅ Saved → {save_dir}/ ({index.ntotal} vectors)\n")

# ─────────────────────────────────────────────────────────────────────────────
# MAIN EXECUTION
# ─────────────────────────────────────────────────────────────────────────────

def ingest(pdf_folder: str, output_dir: str):
    pdf_path_obj = Path(pdf_folder)
    output_path_obj = Path(output_dir)

    pdf_files = list(pdf_path_obj.glob("*.pdf"))
    payer_map = defaultdict(list)
    for pdf in pdf_files:
        payer, disease = extract_payer_disease(pdf.name)
        payer_map[payer].append((pdf, disease))

    for payer in PAYERS:
        print(f"\n{'='*60}")
        print(f"🏛️  PROCESSING PAYER: {payer.upper()}")
        print(f"{'='*60}")

        if payer not in payer_map:
            print(f"  ⚠️  No files found for {payer}. Skipping.")
            continue

        payer_chunks = []
        payer_metadata = []

        for pdf_file, disease in payer_map[payer]:
            print(f"  📄 Reading: {pdf_file.name}")
            text = read_pdf_text(str(pdf_file))
            chunks = chunk_text(text)
            
            for i, chunk in enumerate(chunks):
                payer_chunks.append(chunk)
                payer_metadata.append({
                    "payer": payer, "disease": disease, "source": pdf_file.name, "content": chunk
                })
            print(f"     ∟ Generated {len(chunks)} chunks")

        if payer_chunks:
            build_and_save_faiss(payer_chunks, payer_metadata, output_path_obj / payer)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf_folder", default="./policies")
    parser.add_argument("--output", default="./vector_store")
    args = parser.parse_args()
    ingest(args.pdf_folder, args.output)
# # ─────────────────────────────────────────────────────────────────────────────
# # CONFIG
# # ─────────────────────────────────────────────────────────────────────────────
# # GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
# # genai.configure(api_key=GEMINI_API_KEY)

# # Using the 2026 standard embedding model
# EMBED_MODEL   = "models/gemini-embedding-2-preview"
# CHUNK_SIZE    = 800  # Increased slightly for better context in 2026 models
# CHUNK_OVERLAP = 100

# # The 3 payers — must match your PDF filename prefixes exactly
# PAYERS = ["uhc", "cigna", "aetna"]

# # ─────────────────────────────────────────────────────────────────────────────
# # HELPERS
# # ─────────────────────────────────────────────────────────────────────────────

# def extract_payer_disease(filename: str):
#     """
#     Example: uhc_diabetes_type2_policy.pdf -> payer='uhc', disease='diabetes_type2'
#     """
#     stem = Path(filename).stem.lower()
#     stem = stem.replace("_policy", "")
#     parts = stem.split("_", 1)
#     payer = parts[0]
#     disease = parts[1] if len(parts) > 1 else "unknown"
#     return payer, disease

# def read_pdf_text(path: str) -> str:
#     """Extracts text from PDF with basic cleaning."""
#     try:
#         reader = PdfReader(path)
#         text = ""
#         for page in reader.pages:
#             content = page.extract_text()
#             if content:
#                 text += content + "\n"
#         return text
#     except Exception as e:
#         print(f"  [ERROR] Could not read {path}: {e}")
#         return ""

# def chunk_text(text: str) -> list[str]:
#     """Splits text into overlapping chunks."""
#     chunks = []
#     if not text:
#         return chunks
    
#     start = 0
#     while start < len(text):
#         end = start + CHUNK_SIZE
#         chunks.append(text[start:end])
#         start += CHUNK_SIZE - CHUNK_OVERLAP
#     return chunks

# def get_embeddings_batch(texts: list[str]):
#     """Fetches embeddings using the genai global configuration."""
#     # Use task_type="retrieval_document" for indexing phase
#     result = genai.embed_content(
#         model=EMBED_MODEL,
#         content=texts,
#         task_type="retrieval_document"
#     )
#     return result['embedding']

# def build_and_save_faiss(chunks: list[str], metadata: list[dict], save_dir: Path):
#     """Generates embeddings, builds index, and saves to disk."""
#     save_dir.mkdir(parents=True, exist_ok=True)

#     print(f"  ⏳ Embedding {len(chunks)} chunks (batching for efficiency)...")
    
#     # Batching to stay within API limits and improve speed
#     batch_size = 20
#     all_vectors = []
    
#     for i in range(0, len(chunks), batch_size):
#         batch = chunks[i:i + batch_size]
#         vectors = get_embeddings_batch(batch)
#         all_vectors.extend(vectors)
#         time.sleep(0.5) # Compliance with free tier rate limits

#     matrix = np.array(all_vectors, dtype="float32")
#     dim = matrix.shape[1]

#     # Create and populate FAISS index
#     index = faiss.IndexFlatL2(dim)
#     index.add(matrix)

#     # Save outputs
#     faiss.write_index(index, str(save_dir / "index.faiss"))
#     with open(save_dir / "metadata.pkl", "wb") as f:
#         pickle.dump(metadata, f)

#     print(f"  ✅ Saved → {save_dir}/ ({index.ntotal} vectors)\n")

# # ─────────────────────────────────────────────────────────────────────────────
# # MAIN EXECUTION
# # ─────────────────────────────────────────────────────────────────────────────

# def ingest(pdf_folder: str, output_dir: str):
#     pdf_path_obj = Path(pdf_folder)
#     output_path_obj = Path(output_dir)

#     if not pdf_path_obj.exists():
#         print(f"❌ Error: Folder '{pdf_folder}' not found.")
#         return

#     pdf_files = list(pdf_path_obj.glob("*.pdf"))
#     print(f"\n📂 Ingesting {len(pdf_files)} PDFs from '{pdf_folder}'")

#     # Organize PDFs by payer
#     payer_map = defaultdict(list)
#     for pdf in pdf_files:
#         payer, disease = extract_payer_disease(pdf.name)
#         payer_map[payer].append((pdf, disease))

#     # Process each payer as a separate database
#     for payer in PAYERS:
#         print(f"\n{'='*60}")
#         print(f"🏛️  PROCESSING PAYER: {payer.upper()}")
#         print(f"{'='*60}")

#         if payer not in payer_map:
#             print(f"  ⚠️  No files found for {payer}. Skipping.")
#             continue

#         payer_chunks = []
#         payer_metadata = []

#         for pdf_file, disease in payer_map[payer]:
#             print(f"  📄 Reading: {pdf_file.name}")
#             text = read_pdf_text(str(pdf_file))
#             chunks = chunk_text(text)
            
#             for i, chunk in enumerate(chunks):
#                 payer_chunks.append(chunk)
#                 payer_metadata.append({
#                     "payer": payer,
#                     "disease": disease,
#                     "source": pdf_file.name,
#                     "chunk_id": i,
#                     "content": chunk
#                 })
#             print(f"     ∟ Generated {len(chunks)} chunks")

#         if payer_chunks:
#             dest_folder = output_path_obj / payer
#             build_and_save_faiss(payer_chunks, payer_metadata, dest_folder)

#     print("\n✨ Ingestion Complete. All vector stores are ready.")

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Payer Policy Ingestion Tool")
#     parser.add_argument("--pdf_folder", default="./policies", help="Source folder for PDFs")
#     parser.add_argument("--output", default="./vector_store", help="Root folder for FAISS DBs")
    
#     args = parser.parse_args()
#     ingest(args.pdf_folder, args.output)