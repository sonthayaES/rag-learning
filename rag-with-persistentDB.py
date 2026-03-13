import chromadb
from sentence_transformers import SentenceTransformer
import pypdf 
import re
import os 
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# --- Extract ---
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

# --- Chunk ---
def chunk(text, chunk_size = 500, overlap = 50):
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = words[i: i + chunk_size]
        chunks.append(' '.join(chunk))
        i += chunk_size - overlap
    return chunks


# --- Persistent ChromaDB ---
client = chromadb.PersistentClient(path="./chroma_db")

# Only embed and store if collection doesn't exist yet
existing = [c.name for c in client.list_collections()]

if "bitcoin_docs" not in existing:
    print("First run - embedding and storing chunks...")
    collection = client.create_collection('bitcoin_docs')
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    text = extract_text("data/Bitcoin-Diploma-2025-PDF.pdf")
    chunks = chunk(text)
    embeddings = embed_model.encode(chunks, show_progress_bar = True)
    collection.add(
        documents = chunks,
        embeddings = embeddings.tolist(),
        ids = [f"chunk_{i}" for i in range(len(chunks))]
    )
    print(f'Stored {collection.count()} chunks')
else:
    print("Loading existing collection from disk...")
    collection = client.get_collection("bitcoin_docs")

print(f"Collection has {collection.count()} chunks")

embed_model = SentenceTransformer('all-MiniLM-L6-v2')


# --- Setup DeepSeek ---
deepseek = OpenAI(
    api_key = os.environ.get("DEEPSEEK_API_KEY"),
    base_url = "https://api.deepseek.com"
)

# ---  RAG function with citations ---
def ask(question):
    # Step 1: embed the question
    question_embedding = embed_model.encode(question).tolist()

     # Step 2: retrieve relevant chunks
    results = collection.query(
        query_embeddings = [question_embedding],  # always plural, always a list
        n_results = 5,
        include = ["documents", "distances"]
    )
    retrieved_chunks = results['documents'][0]
    chunk_ids = results['ids'][0]
    distances = results['distances'][0]
    print(f"DEBUG distances for '{question}' : {distances} ")

    
    # Step 3: print sources
    # If all distances are too high (low similarity), don't call LLM at all
    if min(distances) > 1.0:
        print(f"Question: {question}")
        print(f"Answer: This information is not in the provided document.")
        print(f"\nRetrieved chunk IDs: {chunk_ids}")
        print(f"-" *50)
        return
    

    # Step 4: build stricter prompt
    context = ''
    for i, chunk in enumerate(retrieved_chunks):
        context += f"[Score {i+1}]: {chunk}\n\n"

    prompt = f"""You are a helpful assistant that answers questions strictly based on the provided context.

STRICT RULES:
- Answer ONLY using information from the context below
- If the answer is not in the context, respond with exactly: "This information is not in the provided document."
- Do not use any outside knowledge
- At the end of your answer, cite which sources you used like: (Sources: 1, 3)

Context: {context}

Question: {question}

Answer: """
    
    # Step 5: send to DeepSeek
    respond = deepseek.chat.completions.create(
        model = "deepseek-chat",
        messages = [{"role": "user", "content": prompt}]
    )

    answer = respond.choices[0].message.content

     # Step 6: print sources
    print(f"Question: {question}")
    print(f"Answer: {answer}")
    print(f"\nRetrieved chunk IDs: {chunk_ids}")
    print(f"-" *50)

# --- Test it ---
ask("What is the Lightning Network?")
ask("What is the difference between custodial and non-custodial wallets?")
ask("Who is Satoshi Nakamoto?")
ask("What is the capital of France?")  # should say not in document
    

