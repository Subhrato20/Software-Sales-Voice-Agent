import torch
import logging
from transformers import AutoTokenizer, AutoModel, BertForSequenceClassification
from pymilvus import MilvusClient

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Milvus Configuration
MILVUS_URI = "http://localhost:19530"
COLLECTION_NAME = "Software_Sales"

# Initialize Milvus client
client = MilvusClient(uri=MILVUS_URI)

# Load embedding model for search
embedding_tokenizer = AutoTokenizer.from_pretrained('BAAI/bge-large-en-v1.5')
embedding_model = AutoModel.from_pretrained('BAAI/bge-large-en-v1.5')

# Load TinyBERT for re-ranking
reranker_tokenizer = AutoTokenizer.from_pretrained('huawei-noah/TinyBERT_General_6L_768D')
reranker_model = BertForSequenceClassification.from_pretrained('cross-encoder/ms-marco-TinyBERT-L-6')
reranker_model.eval()

def generate_embedding(text):
    """Generates an embedding vector for a given text using BAAI/bge-large-en-v1.5."""
    inputs = embedding_tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
    with torch.no_grad():
        outputs = embedding_model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().tolist()

def search_documents(question, client, collection_name=COLLECTION_NAME, limit=6):
    """Searches for relevant documents in Milvus using embeddings and re-ranks the results."""
    
    query_vector = generate_embedding(question)

    # Perform the search in Milvus without specifying partitions
    try:
        res = client.search(
            collection_name=collection_name,
            data=[query_vector],
            limit=limit,
            output_fields=["text"] 
        )

    except Exception as e:
        logger.error(f"Milvus search failed: {e}")
        return [], []

    # Extract search results including URLs
    search_results = []
    for hit in res[0]:  
        search_results.append({
            "url": hit.get("entity", {}).get("url", ""),
            "text": hit.get("entity", {}).get("text", "")
        })

    if not search_results:
        logger.warning("No relevant documents found.")
        return [], []

    # Re-rank results
    reranked_results = rerank_results(question, search_results)

    return reranked_results


def rerank_results(question, search_results):
    """Re-ranks search results using TinyBERT."""
    scores = []
    for result in search_results:
        candidate_text = result["text"]
        inputs = reranker_tokenizer(question, candidate_text, return_tensors='pt', truncation=True, max_length=512)

        with torch.no_grad():
            logits = reranker_model(**inputs).logits

        score = logits.squeeze().item()
        scores.append((score, result))
    
    # Sort by score (descending)
    sorted_results = sorted(scores, key=lambda x: x[0], reverse=True)
    
    return [item[1] for item in sorted_results]

def search_documents_with_links(query):
    results = search_documents(query, client)
    context = ""

    # Print the top reranked results
    for idx, result in enumerate(results[:4]):
        context += f"<page {idx+1}>\n"
        context += f" page_content:\n {result['text']}\n"
        context += f"</page {idx+1}>\n\n\n\n\n"
    
    return context

if __name__ == "__main__":
    print(search_documents_with_links("tell me about the RAG Services you provide"))
