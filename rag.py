import chromadb
from sentence_transformers import SentenceTransformer
import pypdf
import re
import os
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()

# --- Extract and Chunk ---
def extract_text(pdf_path):
    reader = pypdf.PdfReader(pdf_path)
    full_text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            page_text = re.sub(r'([a-z])([A-Z])', r'\1 \2', page_text)
            page_text = re.sub(r'\s+', ' ', page_text)
            full_text += page_text + "\n"
    return full_text

def chunk_text(text, chunk_size = 500, overlap = 50):
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = words[i: i + chunk_size]
        chunks.append(" ".join(chunk))
        i += chunk_size - overlap
    return chunks

# --- Setup ChromaDB ---
client = chromadb.Client()
collection = client.create_collection("bitcoin_docs")

# --- Embed and store ---
embed_model = SentenceTransformer("all-MiniLm-L6-v2")
text = extract_text("data/Bitcoin-Diploma-2025-PDF.pdf")
chunks = chunk_text(text)

embeddings = embed_model.encode(chunks)

collection.add(
    documents = chunks,
    embeddings = embeddings.tolist(),
    ids = [f"Chunks_{i}" for i in range(len(chunks))]
)

# --- Setup DeepSeek ---
deepseek = OpenAI(
    api_key = os.environ.get("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"
)

# --- The RAG function ---
def ask(question):
    # Step 1: embed the question
    question_embedding = embed_model.encode(question).tolist()

    # Step 2: retrieve relevant chunks
    results = collection.query(
        query_embeddings = [question_embedding],
        n_results = 5
    )
    retrived_chunks = results['documents'][0]

    # Step 3: build the prompt
    context = "\n\n".join(retrived_chunks)
    prompt = f"""You are a helpful assistant. Answer the question using ONLY the context below.
If the answer is not in the context, say "I don't know based on the provided document."

Context: {context}

Question: {question}
Answer:"""
    # Step 4: send to DeepSeek
    respond = deepseek.chat.completions.create(
        model = "deepseek-chat",
        messages = [{"role": "user", "content": prompt}]
    )

    return respond.choices[0].message.content
    
    
# --- Test it ---
questions = [
    "What is the Lightning Network?",
    "What is the difference between custodial and non-custodial wallets?",
    "Who is Satoshi Nakamoto?"
]

for question in questions:
    print(f"\nQ: {question}")
    print(f"A: {ask(question)}")
    print(f"-" * 50)