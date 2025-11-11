import faiss
from sentence_transformers import SentenceTransformer
import numpy as np

# 1. Configuration (using a dense encoder as specified in conversation)
RETRIEVER_MODEL = 'all-mpnet-base-v2'
K = 3 # Number of top documents to retrieve

# 2. Mock Knowledge Base (Non-Parametric Memory)
# This simulates the Wikipedia 100-word chunks used in RAG experiments [2].
KNOWLEDGE_BASE = [
    "The first RAG model was introduced by Patrick Lewis et al. in 2020.",
    "RAG combines a pre-trained seq2seq model (like BART) with a neural retriever (like DPR).",
    "The RAG-Token model can use a different retrieved document for each generated token.",
    "Hallucinations are often defined as factually incorrect or inconsistent outputs.",
    "BART-large is a pre-trained seq2seq transformer with 400M parameters.",
    "A key metric for RAG QA tasks is Exact Match (EM).",
    "RAG is effective for knowledge-intensive NLP tasks."
]

def initialize_retriever_and_index():
    """
    Initializes the dense encoder (retriever) and builds the FAISS MIPS index.
    
    Returns:
        tuple: (SentenceTransformer model, FAISS Index, Knowledge Base List)
    """
    print(f"Initializing Retriever: {RETRIEVER_MODEL}")
    retriever = SentenceTransformer(RETRIEVER_MODEL)

    # Encode all documents in the knowledge base
    doc_embeddings = retriever.encode(KNOWLEDGE_BASE, convert_to_tensor=True).cpu().numpy()
    d = doc_embeddings.shape[3] # Dimension of embeddings

    # Build FAISS index (simulating MIPS for fast nearest neighbor search) [4, 5]
    index = faiss.IndexFlatL2(d)
    index.add(doc_embeddings)
    
    print(f"FAISS Index built with {len(KNOWLEDGE_BASE)} documents.")
    return retriever, index, KNOWLEDGE_BASE

def retrieve_context(query: str, retriever, index) -> (list, str):
    """
    Performs the Maximum Inner Product Search (MIPS) for the query.
    
    Args:
        query (str): The natural language question (x).
        retriever: The initialized SentenceTransformer model.
        index: The initialized FAISS index.

    Returns:
        tuple: (List of retrieved documents (z), Retrieval log string)
    """
    print(f"\n--- Starting Retrieval (Top K={K}) ---")
    
    # Encode the user query (x)
    query_embedding = retriever.encode(query, convert_to_tensor=True).cpu().numpy().reshape(1, -1)

    # Search the top K documents
    D, I = index.search(query_embedding, K) # D=Distances, I=Indices
    
    retrieved_docs = [KNOWLEDGE_BASE[i] for i in I.flatten()]
    
    # Prepare the retrieval log for reporting
    retrieval_log = f"Retrieved {K} documents for query:\n"
    for i, doc in enumerate(retrieved_docs):
        retrieval_log += f"[{i+1}] {doc}\n"
        
    return retrieved_docs, retrieval_log

# If running this file directly for testing:
if __name__ == '__main__':
    retriever_model, faiss_index, kb = initialize_retriever_and_index()
    sample_query = "What LLM generator was used in the first RAG?"
    docs, log = retrieve_context(sample_query, retriever_model, faiss_index)
    print(log)