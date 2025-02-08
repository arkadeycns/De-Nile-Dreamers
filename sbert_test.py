from sentence_transformers import SentenceTransformer

def get_similarity(text1, text2):
    # Load the model
    model = SentenceTransformer("sentence-transformers/paraphrase-distilroberta-base-v1")

    text1_embedded = model.encode(text1, convert_to_numpy=True)
    text2_embedded = model.encode(text2, convert_to_numpy=True)

    # Compute cosine similarity
    similarities = model.similarity(text1_embedded, text2_embedded)
    return similarities[0]
