import chromadb
from sentence_transformers import SentenceTransformer
import pypdf
import re

# --- Step 1: Extract and chunk ---
def extract_text(pdf_path):
    reader = pypdf.PdfReader(pdf_path)
    full_text = ""
    for page in reader.pages:
        page_text = page.extract_text() + "\n"
        if page_text:
            # Fix missing spaces between words
            page_text = re.sub(r'([a-z]) ([A-Z])', r'\1 \2', page_text)
             # Fix multiple spaces/newlines
            page_text = re.sub(r'\s +', ' ' , page_text)
            full_text += page_text + "\n"
    return full_text

def chunk_text(text, chunk_size = 500, overlap = 50):
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = words[i: i+ chunk_size]
        chunks.append(" ".join(chunk))
        i += chunk_size - overlap
    return chunks

# --- Step 2: Setup ChromaDB ---
client = chromadb.Client()
collection = client.create_collection("bitcoin_docs")

# --- Step 3: Embed and store ---
model = SentenceTransformer('all-MiniLM-L6-v2')

text = extract_text("data/Bitcoin-Diploma-2025-PDF.pdf")
chunks = chunk_text(text)

print(f"Embedding {len(chunks)} chunks...")

embeddings = model.encode(chunks, show_progress_bar = True)

collection.add(
    documents = chunks,
    embeddings = embeddings.tolist(),
    ids = [f"chunk_{i}" for i in range(len(chunks))]
)
   
print(f"Stored {collection.count()} chunks in ChromaDB")

# --- Step 4: Query it ---
query = "How do payment channels work in Lightning Network?"
query_embedding = model.encode(query).tolist()

results = collection.query(
    query_embeddings = [query_embedding],
    n_results = 5
)

print(f"\nQuery: '{query}'")
print(f"\n--- Top 5 query ---")
for i, doc in enumerate(results['documents'][0]):
    print(f"\nResult: {i+1}:\n{doc[:300]}")
    print(f"...")