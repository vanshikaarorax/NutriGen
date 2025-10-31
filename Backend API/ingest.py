
import os, uuid, re
import fitz  # PyMuPDF
import tiktoken
from supabase import create_client, Client
from openai import OpenAI
from tqdm import tqdm
from dotenv import load_dotenv, find_dotenv

# Load environment
load_dotenv(find_dotenv(usecwd=True))

SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_SERVICE_ROLE_KEY = os.environ["SUPABASE_SERVICE_ROLE_KEY"]
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]

# Config
PDF_PATH = "Dataset-nutrition.pdf"
DOC_ID = "nutrition-v1"  # keep this STABLE to avoid duplicates
EMBED_MODEL = "text-embedding-3-small"  # 1536 in your table
BATCH_EMBED = 100
BATCH_INSERT = 200

# Sentence chunking params
SENTS_PER_CHUNK = 20
SENT_OVERLAP = 2
MAX_TOKENS = 1300  # safety cap (trim if 10 sentences are too long)
MIN_TOKENS = 50  # skip very tiny fragments

enc = tiktoken.get_encoding("cl100k_base")  # matches OpenAI embeddings tokenizer


def clean_text(t: str) -> str:
    """Normalize whitespace and fix hyphenation across line breaks."""
    t = t.replace("\r", "")
    t = re.sub(r"-\s*\n\s*", "", t)  # join "nutri-\n tion" => "nutrition"
    t = re.sub(r"\s+\n", "\n", t)
    t = re.sub(r"[ \t]+", " ", t)
    t = t.replace("\n", " ").strip()
    return t


def split_sentences(text: str):
    """Split text into sentences based on punctuation."""
    sents = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sents if s.strip()]


def chunk_page_by_sentences(
    text: str,
    sents_per_chunk: int = SENTS_PER_CHUNK,
    overlap: int = SENT_OVERLAP,
    max_tokens: int = MAX_TOKENS,
    min_tokens: int = MIN_TOKENS
):
    """Yield chunks of sentences within token limits."""
    sents = split_sentences(text)
    i = 0
    step = max(1, sents_per_chunk - overlap)

    while i < len(sents):
        piece = sents[i:i + sents_per_chunk]
        if not piece:
            break

        chunk = " ".join(piece)
        ids = enc.encode(chunk)

        # trim if chunk exceeds max_tokens
        while max_tokens and len(ids) > max_tokens and len(piece) > 1:
            piece = piece[:-1]
            chunk = " ".join(piece)
            ids = enc.encode(chunk)

        if len(ids) >= min_tokens:
            yield chunk

        i += step


def pdf_pages(path: str):
    """Yield (page_number_1based, cleaned_text)."""
    doc = fitz.open(path)
    try:
        for i in range(len(doc)):
            txt = doc[i].get_text("text") or ""
            yield (i + 1, clean_text(txt))
    finally:
        doc.close()


def main():
    sb: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)
    client = OpenAI(api_key=OPENAI_API_KEY)

    # Optional: clean up existing entries for this document
    sb.table("chunks").delete().eq("doc_id", DOC_ID).execute()

    print("Reading PDF by pages ...")
    pages = list(pdf_pages(PDF_PATH))

    # Build chunks with page metadata
    inputs, metas = [], []
    print(f"Chunking ({SENTS_PER_CHUNK} sentences per chunk, {SENT_OVERLAP} overlap) ...")

    for page, text in pages:
        if not text:
            continue
        for chunk in chunk_page_by_sentences(text):
            inputs.append(chunk)
            metas.append({"page": page, "source": PDF_PATH})

    print(f"Built {len(inputs)} chunks from {PDF_PATH}")

    # Generate embeddings and upload to Supabase
    print("Generating embeddings and uploading to Supabase...")

    for i in tqdm(range(0, len(inputs), BATCH_EMBED)):
        batch = inputs[i:i + BATCH_EMBED]
        meta_batch = metas[i:i + BATCH_EMBED]

        # Generate embeddings in batch
        response = client.embeddings.create(
            model=EMBED_MODEL,
            input=batch
        )

        embeddings = [item.embedding for item in response.data]

        # Prepare rows for insertion
        rows = []
        for text_chunk, emb, meta in zip(batch, embeddings, meta_batch):
            rows.append({
    "doc_id": DOC_ID,
    "chunk_index": metas.index(meta),
    "content": text_chunk,
    "metadata": meta,
    "embedding": emb
})


        # Batch insert into Supabase
        for j in range(0, len(rows), BATCH_INSERT):
            sb.table("chunks").insert(rows[j:j + BATCH_INSERT]).execute()

    print("✅ All chunks embedded and uploaded successfully!")


if __name__ == "__main__":
    main()
