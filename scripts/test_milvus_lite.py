import os
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../")

from config.storage_config import MILVUS_DATA_PATH
from pymilvus import MilvusClient

def test_milvus_lite():
    print(f"Testing Milvus Lite at: {MILVUS_DATA_PATH}")
    
    # Clean up previous test
    if os.path.exists(MILVUS_DATA_PATH):
        try:
            # We can't easily delete the file if it's locked, but let's try to connect
            pass
        except:
            pass

    try:
        client = MilvusClient(uri=MILVUS_DATA_PATH)
        print("Successfully connected to Milvus Lite.")
        
        col_name = "test_collection"
        if client.has_collection(col_name):
            client.drop_collection(col_name)
            
        client.create_collection(
            collection_name=col_name,
            dimension=4
        )
        print(f"Created collection: {col_name}")
        
        data = [
            {"id": 1, "vector": [0.1, 0.2, 0.3, 0.4], "text": "hello"},
            {"id": 2, "vector": [0.4, 0.3, 0.2, 0.1], "text": "world"}
        ]
        
        client.insert(collection_name=col_name, data=data)
        print("Inserted data.")
        
        res = client.search(
            collection_name=col_name,
            data=[[0.1, 0.2, 0.3, 0.4]],
            limit=1,
            output_fields=["text"]
        )
        print(f"Search result: {res}")
        
        client.drop_collection(col_name)
        print("Test passed.")
        
    except Exception as e:
        print(f"Test failed: {e}")
        raise e

if __name__ == "__main__":
    test_milvus_lite()
