from rag_chunking_pipeline import process_support_articles
from vector_store import ChromaVectorStore
import json
import os
from typing import List, Dict

def load_support_articles(path: str) -> List[Dict]:
    """
    Loads support articles from a json file or directory.
    Each article must contain: id, text, category, priority, last_updated_date.
    """
    if os.path.isdir(path):
        articles = []
        for fname in os.listdir(path):
            if fname.endswith('.json'):
                with open(os.path.join(path, fname), 'r', encoding='utf-8') as f:
                    articles.append(json.load(f))
        return articles
    else:
        with open(path, 'r', encoding='utf-8') as f:
            articles = json.load(f)
        return articles

def populate(
    articles_path: str,
    persist_directory: str,
    collection_name: str,
    embedding_model_name: str = 'sentence-transformers/all-MiniLM-L6-v2'
):
    articles = load_support_articles(articles_path)
    processed_chunks = process_support_articles(articles, chunk_size=500, overlap=200)
    vs = ChromaVectorStore(persist_directory, collection_name, embedding_model_name)
    vs.add_chunks(processed_chunks)
    print(f"Populated Chroma vector store with {len(processed_chunks)} chunks.")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--articles_path', type=str, required=True, help='Path to articles json or directory')
    parser.add_argument('--persist_directory', type=str, default='chromadb_data')
    parser.add_argument('--collection_name', type=str, default='support_articles')
    parser.add_argument('--embedding_model_name', type=str, default='sentence-transformers/all-MiniLM-L6-v2')
    args = parser.parse_args()
    populate(
        args.articles_path,
        args.persist_directory,
        args.collection_name,
        args.embedding_model_name
    )
