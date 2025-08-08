import re
from typing import List, Dict

def tokenize_text(text: str) -> List[str]:
    """
    Simple whitespace-based tokenization.
    For production, use a tokenizer matching the embedding model's tokenizer.
    """
    tokens = text.split()
    return tokens

def detokenize(tokens: List[str]) -> str:
    return ' '.join(tokens)

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 200) -> List[str]:
    """
    Splits text into token chunks of chunk_size with given overlap.
    """
    tokens = tokenize_text(text)
    n = len(tokens)
    chunks = []
    i = 0
    while i < n:
        chunk_tokens = tokens[i:i+chunk_size]
        chunk_text = detokenize(chunk_tokens)
        chunks.append(chunk_text)
        if i + chunk_size >= n:
            break
        i += chunk_size - overlap
    return chunks

def process_support_articles(articles: List[Dict], chunk_size: int = 500, overlap: int = 200) -> List[Dict]:
    """
    For each article, chunk the text and attach metadata to each chunk.
    Each article dict expected to have: 'id', 'text', 'category', 'priority', 'last_updated_date'.
    Returns a list of dicts with 'id', 'text', 'metadata'.
    """
    processed_chunks = []
    for article in articles:
        chunks = chunk_text(article['text'], chunk_size, overlap)
        for idx, chunk in enumerate(chunks):
            chunk_id = f"{article['id']}_chunk_{idx}"
            metadata = {
                'category': article['category'],
                'priority': article['priority'],
                'last_updated_date': article['last_updated_date'],
                'source_id': article['id'],
                'chunk_index': idx
            }
            processed_chunks.append({
                'id': chunk_id,
                'text': chunk,
                'metadata': metadata
            })
    return processed_chunks
