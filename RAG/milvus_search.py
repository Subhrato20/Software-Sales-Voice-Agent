# src/agent/tools/rag_handler.py
import torch
import logging
from transformers import AutoTokenizer, AutoModel, BertForSequenceClassification
from pymilvus import MilvusClient
import os
from dotenv import load_dotenv

# Load environment variables if you use a .env file for configurations
load_dotenv()

# Configure logging
# The main server.py or your application entry point should ideally configure logging globally.
# For standalone testing of this module, basicConfig can be used.
# logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class RAGHandler:
    def __init__(self):
        # Milvus Configuration
        self.MILVUS_URI = os.getenv("MILVUS_URI", "http://localhost:19530")
        self.COLLECTION_NAME = os.getenv("RAG_COLLECTION_NAME", "Software_Sales")

        logger.info(f"RAGHandler: Initializing Milvus client with URI: {self.MILVUS_URI} and Collection: {self.COLLECTION_NAME}")
        try:
            self.client = MilvusClient(uri=self.MILVUS_URI)
            if not self.client.has_collection(collection_name=self.COLLECTION_NAME):
                 logger.warning(f"RAGHandler: Milvus collection '{self.COLLECTION_NAME}' not found at {self.MILVUS_URI}. Search may fail or return no results.")
            else:
                 logger.info(f"RAGHandler: Successfully connected to Milvus and found collection '{self.COLLECTION_NAME}'.")
        except Exception as e:
            logger.error(f"RAGHandler: Failed to initialize Milvus client or check collection. URI: {self.MILVUS_URI}, Collection: {self.COLLECTION_NAME}. Error: {e}")
            self.client = None

        # Load models - consider making model names configurable via environment variables
        self.embedding_tokenizer_name = "BAAI/bge-large-en-v1.5"
        self.embedding_model_name = "BAAI/bge-large-en-v1.5"
        # The original TinyBERT_General_6L_768D is a base model. For reranking, a cross-encoder is typically used.
        # The cross-encoder/ms-marco-TinyBERT-L-6 is appropriate.
        self.reranker_tokenizer_name = 'huawei-noah/TinyBERT_General_6L_768D' # Tokenizer for the cross-encoder often matches its base
        self.reranker_model_name = 'cross-encoder/ms-marco-TinyBERT-L-6'

        try:
            logger.info(f"RAGHandler: Loading embedding tokenizer: {self.embedding_tokenizer_name}")
            self.embedding_tokenizer = AutoTokenizer.from_pretrained(self.embedding_tokenizer_name)
            logger.info(f"RAGHandler: Loading embedding model: {self.embedding_model_name}")
            self.embedding_model = AutoModel.from_pretrained(self.embedding_model_name)

            logger.info(f"RAGHandler: Loading reranker tokenizer: {self.reranker_tokenizer_name}")
            self.reranker_tokenizer = AutoTokenizer.from_pretrained(self.reranker_tokenizer_name)
            logger.info(f"RAGHandler: Loading reranker model: {self.reranker_model_name}")
            self.reranker_model = BertForSequenceClassification.from_pretrained(self.reranker_model_name)
            self.reranker_model.eval()
            logger.info("RAGHandler: All models loaded successfully.")
        except Exception as e:
            logger.error(f"RAGHandler: Error loading models: {e}")
            # Potentially set flags to indicate models are not loaded, or re-raise
            raise RuntimeError(f"Failed to load HuggingFace models: {e}")

        logger.info("RAGHandler initialized.")

    def _generate_embedding(self, text: str) -> list:
        """Generates an embedding vector for a given text."""
        try:
            inputs = self.embedding_tokenizer(text, return_tensors='pt', truncation=True, max_length=512, padding=True)
            with torch.no_grad():
                outputs = self.embedding_model(**inputs)
            return outputs.last_hidden_state.mean(dim=1).squeeze().tolist()
        except Exception as e:
            logger.error(f"Error generating embedding for text '{text[:50]}...': {e}")
            return [] # Return empty list on failure

    def _rerank_results(self, question: str, search_results: list) -> list:
        """Re-ranks search results using TinyBERT."""
        if not search_results:
            return []
        
        scores = []
        for result in search_results:
            candidate_text = result.get("text", "")
            if not candidate_text:
                scores.append((-float('inf'), result)) # Low score for empty text
                continue
            try:
                inputs = self.reranker_tokenizer(question, candidate_text, return_tensors='pt', truncation=True, max_length=512, padding=True)
                with torch.no_grad():
                    logits = self.reranker_model(**inputs).logits
                score = logits.squeeze().item()
                scores.append((score, result))
            except Exception as e:
                logger.error(f"Error reranking result for query '{question[:50]}...' with text '{candidate_text[:50]}...': {e}")
                scores.append((-float('inf'), result)) # Low score on error
        
        sorted_results = sorted(scores, key=lambda x: x[0], reverse=True)
        return [item[1] for item in sorted_results]

    def _search_milvus_documents(self, question: str, limit: int = 6) -> list:
        """Searches for relevant documents in Milvus using embeddings."""
        if not self.client:
            logger.error("RAGHandler: Milvus client not initialized. Cannot perform search.")
            return []

        query_vector = self._generate_embedding(question)
        if not query_vector: # Check if embedding generation failed
            logger.error(f"RAGHandler: Failed to generate query vector for: {question[:50]}")
            return []
        
        search_results_list = []
        try:
            logger.info(f"RAGHandler: Searching Milvus collection '{self.COLLECTION_NAME}' for query: {question[:50]}...")
            # Assuming 'text' and 'url' are the fields you stored and want to retrieve
            milvus_response = self.client.search(
                collection_name=self.COLLECTION_NAME,
                data=[query_vector],
                limit=limit,
                output_fields=["text", "url"] 
            )
            
            if milvus_response and milvus_response[0]:
                for hit in milvus_response[0]:
                    entity = hit.get("entity", {})
                    search_results_list.append({
                        "url": entity.get("url", "URL not available"),
                        "text": entity.get("text", "Text not available")
                    })
            else:
                logger.warning(f"RAGHandler: No hits found in Milvus for query: {question[:50]}")

        except Exception as e:
            logger.error(f"RAGHandler: Milvus search failed for collection '{self.COLLECTION_NAME}': {e}")
            return [] # Return empty list on Milvus error
            
        return search_results_list

    def search_documents_with_links(self, query: str) -> str:
        """
        User-provided function: Searches documents, re-ranks, and formats them as a string.
        Logic remains as provided by the user, integrated into the class structure.
        """
        if not self.client:
            logger.error("RAGHandler: Cannot search, Milvus client is not available.")
            return "Error: The knowledge base is currently unavailable. Please try again later."

        logger.info(f"RAGHandler: Received query for RAG: {query}")
        
        raw_search_results = self._search_milvus_documents(query)
        if not raw_search_results:
            logger.info(f"RAGHandler: No initial documents found for query: {query[:50]}")
            return "I couldn't find any documents related to your query in the knowledge base."

        reranked_results = self._rerank_results(query, raw_search_results)
        if not reranked_results:
            logger.info(f"RAGHandler: No documents remained after reranking for query: {query[:50]}")
            return "I found some initial information, but couldn't refine it to a specific answer for your query."

        context = ""
        # Format the top 4 reranked results as in the original snippet
        for idx, result in enumerate(reranked_results[:4]):
            context += f"<page {idx+1}>\n"
            page_content = result.get('text', 'Content not available.')
            context += f"page_content:\n{page_content}\n"
            # Optional: include URL if desired and available
            # page_url = result.get('url')
            # if page_url and page_url != "URL not available":
            #     context += f"source_url: {page_url}\n"
            context += f"</page {idx+1}>\n\n"
        
        if not context.strip():
             logger.info(f"RAGHandler: Context is empty after formatting results for query: {query[:50]}")
             return "I found some documents, but could not extract the specific content needed to answer your query."

        logger.info(f"RAGHandler: Returning context with {len(reranked_results[:4])} chunks for query: {query[:50]}")
        return context.strip()

# if __name__ == "__main__":
#     # This section is for standalone testing of the RAGHandler
#     logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - [%(name)s] - %(message)s")
#     # Ensure MILVUS_URI and RAG_COLLECTION_NAME are set in your .env or environment
#     # Also ensure your Milvus instance is running and the collection exists and is loaded with data.
#     try:
#         rag_handler = RAGHandler()
#         if rag_handler.client: # Proceed only if client initialized
#             test_query = "tell me about the RAG Services you provide"
#             logger.info(f"Main: Testing RAG with query: '{test_query}'")
#             result_context = rag_handler.search_documents_with_links(test_query)
#             print("\nReturned Context:\n", result_context)
#         else:
#             logger.error("Main: RAGHandler client failed to initialize. Cannot run test.")
#     except RuntimeError as e:
#         logger.error(f"Main: Failed to initialize RAGHandler: {e}")
#     except Exception as e:
#         logger.error(f"Main: An unexpected error occurred: {e}")