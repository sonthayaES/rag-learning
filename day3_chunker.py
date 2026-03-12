import pypdf

def extract_text(pdf_path):
    reader = pypdf.PdfReader(pdf_path)
    full_text = ""
    for page in reader.pages:
        full_text += page.extract_text() + "\n"
    return full_text

def chunk_text(text, chunk_size=500, overlap=50):
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = words[i:i + chunk_size]
        chunks.append(" ".join(chunk))
        i += chunk_size - overlap
    return chunks

# Run it
text = extract_text("data/Bitcoin-Diploma-2025-PDF.pdf")
chunks = chunk_text(text)

print(f"Total characters extracted: {len(text)}")
print(f"Total chunks created: {len(chunks)}")
print(f"--- Chunk1 --- {chunks[0]}")
print(f"--- Chunk2 --- {chunks[1]}")

for i, chunk in enumerate(chunks[5]):
    print(f"\n --- Chunk {i+1} ({len(chunk.split())} words) ---")
    print("...")

garbage_chunks = [c for c in chunks if len(c.split()) < 10]
good_chunks = [c for c in chunks if len(c.split()) > 10]
print(f"Total Chunks: {len(chunks)}")
print(f"Garbage Chunks (< 10 words) {len(garbage_chunks)}")
print(f"Good Chunks: (10+ words){len(good_chunks)}")
print(f"\nSample Chunks")
for c in garbage_chunks[:5]:
    print(f"  '{c}'")
