# The Big Picture
-This repo is build system to read PDF file, understanding it  meaning, and answer questions about "Document we use" with LLM - without LLM to train on my Document.

# Flow of system
PDF -> extract -> chunk -> embed -> store -> query -> retrieve -> LLM -> answer.

Day1 - Concept
- Embeddings = the way to turn text to a value in vector.
- Meaning is learned from content patterns across sentences - Not rule or dictionaries.
- Cosine similarity: 1.0 = identical and 0 = unrelated.
- RAG flow: Document -> chunk -> embeddings -> vector DB -> retrieve -> LLM.

Day2 - First code
- (`model.encode(sentence)`) converts text into a vector.
- Cosine similarity formula:(`np.dot(a,b) / (norm(a) * norm(b))`).
- (`enumerate()`) gives index + item in a loop.
- (`f"{score:.4f}"`) formats floats to 4 decimal places.

Day 3 — Chunking
- (`pypdf.PdfReader()`) read PDFs, (`.pages`) give all pages.
- (`+=`) accumulates, (`=`) overwrites -** Critical different in loop**
- (`text.split()`) splits by whitespace into word list.
- (`words[i: i + 500]`) slices 500 words starting at position i.
- (`" ".join(chunk)`) resembles words back into readable text.
- Overlap prevents concepts from being cut at chunk boundaries.
- List comprehension: (`[x for x in list if condition]`).

Day 4 — Vector Database
- (`chromadb.Client()`) = in memory, lost when script stops.
- (`chromadb.PersistentClient(path="./chroma_db")`) = saved to disk.
- (`collection.add(documents, embeddings, ids)`) stores chunks.
- (`collection.query(query_embeddings, n_results)`) finds similar chunks.
- (`include=["documents", "distances"]`) — ids excluded because always   returned automatically.
- ChromaDB always returns n_results chunks even if none are relevant (forced retrieval).

Day 5 — DeepSeek Integration
- DeepSeek uses same (`openai`) Python library, just different (`base_url`).
- Prompt = context + question + strict rules.
- RAG = retrieval augmented generation — giving LLM your document as context at query time.
- Prompt leakage = LLM ignores instructions and answers from its own training knowledge.

Day 6 — Production Improvements
- (`env`) file stores API keys safely, never commit to GitHub.
- (`load_dotenv()`) loads (`.env`) into (`os.environ`) automatically.
- Distance threshold blocks irrelevant questions before hitting LLM.
- Lower distance = more similar (0.675 very relevant, 1.765 completely irrelevant).
- Tuning threshold requires measuring real distances, not guessing
- Source citations show which chunks produced each answer

# My cheat sheet
(`
# Embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')
embedding = model.encode("sentence").tolist()  # .tolist() for ChromaDB

# ChromaDB setup
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.create_collection("name")
collection = client.get_collection("name")
existing = [c.name for c in client.list_collections()]

# ChromaDB store
collection.add(documents=chunks, embeddings=embeddings, ids=ids)

# ChromaDB query
results = collection.query(
    query_embeddings=[embedding],   # plural, list
    n_results=5,
    include=["documents", "distances"]  # no "ids" here
)
chunks = results['documents'][0]
distances = results['distances'][0]
ids = results['ids'][0]  # ids always available even without include

# DeepSeek
from openai import OpenAI
client = OpenAI(api_key=os.environ.get("KEY"), base_url="https://api.deepseek.com")
response = client.chat.completions.create(
    model="deepseek-chat",
    messages=[{"role": "user", "content": prompt}]
)
answer = response.choices[0].message.content

# .env
from dotenv import load_dotenv
load_dotenv()  # must call before os.environ.get()
`)
