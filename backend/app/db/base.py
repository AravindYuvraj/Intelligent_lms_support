from pymongo import MongoClient
import redis
from backend.app.core.config import settings

# MongoDB Database
mongodb_client = MongoClient(settings.MONGODB_URL)
mongodb_db = mongodb_client["lms_support"]

# Redis Client
redis_client = redis.from_url(settings.REDIS_URL)

def get_mongodb():
    return mongodb_db

def get_redis():
    return redis_client