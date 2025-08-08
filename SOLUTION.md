# Solution Steps

1. Create a chunking pipeline (rag_chunking_pipeline.py): Implement token-based chunking for articles with chunk size 500 and overlap 200. Attach metadata (category, priority, last_updated_date, plus source_id and chunk_index) to each chunk.

2. Implement the vector store interface (vector_store.py): Use ChromaDB and sentence-transformers for embedding. Provide methods to add processed chunks with metadata, and perform cosine similarity top-5 retrieval.

3. Build a population script (populate_vector_store.py): Load articles (from JSON file or directory), process/chunk them, and add their embeddings and metadata into the Chroma vector store.

4. Create an answer generation script (generate.py): For a given query, retrieve the top-5 relevant chunks (text and metadata), build a context, and pass it to the LLM (e.g., OpenAI GPT-3.5) to generate a grounded answer.

5. Ensure sentence-transformers is used for embeddings and Chroma for storage. Verify top-5 retrieval and that metadata is consistently attached and available for downstream usage.

6. Test with sample/reasonable article data and user queries to confirm correct chunking, population, retrieval, and generation pipeline behavior.

