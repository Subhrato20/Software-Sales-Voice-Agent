import os
import json
import logging
from pymilvus import MilvusClient, MilvusException

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Milvus connection settings
MILVUS_URI = "http://localhost:19530"
COLLECTION_NAME = "Software_Sales"

# Initialize Milvus client
client = MilvusClient(uri=MILVUS_URI)

def load_json(file_path: str):
    """Load data from a JSON file. Returns an empty list on failure."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading JSON file {file_path}: {e}")
        return []

def upsert_to_milvus(client: MilvusClient, collection_name: str, json_objects: list):
    """Upserts data into Milvus collection (no partition)."""
    if not json_objects:
        logger.error(f"No data found to upsert for collection '{collection_name}'.")
        return

    try:
        client.upsert(
            collection_name=collection_name,
            data=json_objects
        )
        logger.info(f"Upserted {len(json_objects)} records into collection '{collection_name}'.")
    except MilvusException as e:
        logger.error(f"Failed to upsert data into '{collection_name}': {e}")

def process_json_files(json_directory: str):
    """Process and upsert each JSON file in the given directory."""
    for file_name in os.listdir(json_directory):
        if not file_name.endswith(".json"):
            continue

        file_path = os.path.join(json_directory, file_name)
        json_data = load_json(file_path)

        # Upsert directly into the single collection
        upsert_to_milvus(client, COLLECTION_NAME, json_data)

if __name__ == "__main__":
    json_directory = "../processed_data"
    if not os.path.isdir(json_directory):
        logger.error(f"JSON directory '{json_directory}' does not exist.")
    else:
        process_json_files(json_directory)
