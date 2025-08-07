# history.py

import pandas as pd
from pymongo import MongoClient

# MongoDB Config (can be parameterized)
MONGO_URI = "mongodb://localhost:27017"
DB_NAME = "object_detection"

client = MongoClient(MONGO_URI)
db = client[DB_NAME]

def get_all_collections():
    return db.list_collection_names()

def get_detection_history(user_id, class_filter=None, source_filter=None, limit=100):
    results = []
    for col_name in get_all_collections():
        if col_name == "users":
            continue
        collection = db[col_name]
        query = {"user_id": user_id}
        if class_filter:
            query["object_counts." + class_filter] = {"$exists": True}
        if source_filter:
            query["source"] = source_filter
        docs = collection.find(query).sort("timestamp", -1).limit(limit)
        for doc in docs:
            doc["collection"] = col_name
            results.append(doc)
    return results

def history_to_dataframe(docs):
    rows = []
    for doc in docs:
        timestamp = doc.get("timestamp")
        model_type = doc.get("model_type")
        source = doc.get("source")
        image_name = doc.get("image_name")
        collection = doc.get("collection", "")
        for cls, count in doc.get("object_counts", {}).items():
            rows.append({
                "Date": timestamp,
                "Class": cls,
                "Count": count,
                "Model": model_type,
                "Source": source,
                "Image": image_name,
                "Collection": collection
            })
    return pd.DataFrame(rows)
