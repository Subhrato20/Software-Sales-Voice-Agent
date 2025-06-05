from pymilvus import MilvusClient

MILVUS_URI = "http://localhost:19530"
COLLECTION_NAME = "Software_Sales"

client = MilvusClient(uri=MILVUS_URI)
client.drop_collection(collection_name=COLLECTION_NAME)
