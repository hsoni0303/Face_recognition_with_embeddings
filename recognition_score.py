from sklearn.metrics.pairwise import cosine_similarity

def calculate_cosine_similarity(embedding_1, embedding_2):
  similarities = cosine_similarity(embedding_1, embedding_2)
  return similarities