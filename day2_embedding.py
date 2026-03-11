from sentence_transformers import SentenceTransformer
import numpy as np

# Load the model (downloads ~80MB first time, then cached)
model = SentenceTransformer('all-MiniLM-L6-v2')


# Embed a sentence
sentences = [
    "I love dog",
    "I adore puppies",
    "The shock market crashed"
]


embedding = model.encode(sentences)


def consine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

base = embedding[0] #I love dog

for i, sentence in enumerate(sentences):
    score = consine_similarity(base, embedding[i])
    print(f"'{sentences[0]}' vs '{sentence}' -> '{score:.4f}'")
   
print(f"\nVector length: {len(embedding[0])}")
print(f"First 10 numbers: {embedding[0][:10]}")



