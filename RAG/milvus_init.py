from pymilvus import MilvusClient
from pymilvus.exceptions import MilvusException

import logging

# Configure logger
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

MILVUS_URI = "http://localhost:19530"

COLLECTIONS_CONFIG = {
    "Software_Sales": {
        "dimension": 1024,
        "metric_type": "COSINE"
    }
}  # type: ignore

# Initialize Milvus Client
client = MilvusClient(uri=MILVUS_URI)

def get_milvus_client():
    return client

def create_collection(client: MilvusClient, name: str, dimension: int, metric_type: str) -> None:
    """Creates a Milvus collection."""
    try:
        client.create_collection(
            collection_name=name,
            dimension=dimension,
            metric_type=metric_type
        )
        logger.info(f"Collection '{name}' created successfully.")
    except MilvusException as e:
        if "already exists" in str(e):
            logger.warning(f"Collection '{name}' already exists.")
        else:
            logger.error(f"Failed to create collection '{name}': {e}")
            raise

def main():
    # Initialize Milvus client
    client = MilvusClient(uri=MILVUS_URI)
    logger.info("Connected to Milvus.")

    # Create collections
    for collection_name, details in COLLECTIONS_CONFIG.items():
        create_collection(
            client,
            collection_name,
            details["dimension"],
            details["metric_type"]
        )

if __name__ == "__main__":
    main()
