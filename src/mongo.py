from pymongo import MongoClient

# MongoDB Configuration
MONGO_URI = "mongodb://localhost:27017"  # or replace with your Atlas URI
DB_NAME = "object_detection"


# Connect to MongoDB
client = MongoClient(MONGO_URI)
db = client[DB_NAME]
USERS_COLLECTION = "users"
users_collection = db[USERS_COLLECTION]