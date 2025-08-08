from chromadb import Client
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from typing import List, Dict

class ChromaVectorStore:
    def __init__(self, persist_directory: str, collection_name: str, embedding_model_name: str = 'sentence-transformers/all-MiniLM-L6-v2'):
        self.client = Client(Settings(persist_directory=persist_directory))
        self.collection_name = collection_name
        self.embedder = SentenceTransformer(embedding_model_name)
        if collection_name in [c.name for c in self.client.list_collections()]:
            self.collection = self.client.get_collection(collection_name)
        else:
            self.collection = self.client.create_collection(collection_name)

    def add_chunks(self, chunks: List[Dict]):
        # Prepare data
        ids = [chunk['id'] for chunk in chunks]
        texts = [chunk['text'] for chunk in chunks]
        embeddings = self.embedder.encode(texts, show_progress_bar=True, convert_to_numpy=True)
        metadatas = [chunk['metadata'] for chunk in chunks]
        self.collection.add(
            documents=texts,
            embeddings=embeddings,
            ids=ids,
            metadatas=metadatas
        )

    def retrieve(self, query: str, top_k: int = 5):
        query_embedding = self.embedder.encode([query], convert_to_numpy=True)[0]
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=['documents', 'metadatas', 'ids']
        )
        hits = []
        for i in range(len(results['ids'][0])):
            hits.append({
                'id': results['ids'][0][i],
                'text': results['documents'][0][i],
                'metadata': results['metadatas'][0][i]
            })
        return hits
