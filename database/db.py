from typing import List
from pymongo import MongoClient
from bson import ObjectId
import os
from config import env

try:
    client = MongoClient(env.MONGO_URI)
    db = client["paper_checking"]
    print("Connected to MongoDB!")
except Exception as e:
    print(f"Error connecting to MongoDB: {e}")

collection = db["mcq_results"]
mcq_collection = db["reference_mcqs"]
subjectivePaper_collection = db["subjective_papers"]
jobs_collection = db["jobs"]
# In-memory storage (alternative: use Redis in production)
teacher_reference_data = {}
image_queue: List[bytes] = []
processing_active = False