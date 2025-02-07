from sentence_transformers import SentenceTransformer
model = SentenceTransformer("sentence-transformers/paraphrase-MiniLM-L6-v2")
text1 = input()
text2 = input()
text1_embedded = model.encode(text1)
text2_embedded = model.encode(text2)

similarities = model.similarity(text1_embedded, text2_embedded)
print(similarities)