import json
import numpy as np
from typing import Dict, Any, Optional, List
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from backend.app.db.base import get_redis
from backend.app.core.config import settings
import logging

logger = logging.getLogger(__name__)

class SemanticCacheService:
    def __init__(self):
        self.redis_client = get_redis()
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-001",
            google_api_key=settings.GOOGLE_API_KEY
        )
    
    async def search_similar(self, query: str, threshold: float = 0.85) -> Optional[Dict[str, Any]]:
        """Search for semantically similar queries in cache"""
        try:
            # Generate embedding for the query
            query_embedding = await self.embeddings.aembed_query(query)
            query_vector = np.array(query_embedding)
            
            # Get all cached queries
            cached_keys = self.redis_client.keys("cache:*")
            
            best_match = None
            best_similarity = 0.0
            
            for key in cached_keys:
                try:
                    cached_data = json.loads(self.redis_client.get(key))
                    cached_embedding = np.array(cached_data["embedding"])
                    
                    # Calculate cosine similarity
                    similarity = np.dot(query_vector, cached_embedding) / (
                        np.linalg.norm(query_vector) * np.linalg.norm(cached_embedding)
                    )
                    
                    if similarity > best_similarity and similarity >= threshold:
                        best_similarity = similarity
                        best_match = cached_data
                        
                except Exception as e:
                    logger.error(f"Error processing cached item {key}: {str(e)}")
                    continue
            
            if best_match:
                logger.info(f"Found cached response with similarity: {best_similarity}")
                return {
                    "response": best_match["response"],
                    "confidence": best_match.get("confidence", 0.9),
                    "similarity": best_similarity,
                    "original_query": best_match["query"]
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Semantic cache search error: {str(e)}")
            return None
    
    async def store_response(
        self, 
        query: str, 
        response: str, 
        confidence: float,
        category: str,
        metadata: Dict[str, Any] = None
    ):
        """Store a query-response pair in semantic cache"""
        try:
            # Generate embedding for the query
            query_embedding = await self.embeddings.aembed_query(query)
            
            # Create cache entry
            cache_data = {
                "query": query,
                "response": response,
                "confidence": confidence,
                "category": category,
                "embedding": query_embedding,
                "metadata": metadata or {},
                "timestamp": int(np.datetime64('now').astype(np.int64) / 1e9)
            }
            
            # Store in Redis with expiration (7 days)
            cache_key = f"cache:{hash(query)}"
            self.redis_client.setex(
                cache_key,
                604800,  # 7 days in seconds
                json.dumps(cache_data)
            )
            
            logger.info(f"Stored response in cache for query: {query[:50]}...")
            
        except Exception as e:
            logger.error(f"Error storing in cache: {str(e)}")
    
    async def invalidate_category(self, category: str):
        """Invalidate cache entries for a specific category (useful when knowledge base is updated)"""
        try:
            cached_keys = self.redis_client.keys("cache:*")
            
            for key in cached_keys:
                try:
                    cached_data = json.loads(self.redis_client.get(key))
                    if cached_data.get("category") == category:
                        self.redis_client.delete(key)
                        logger.info(f"Invalidated cache entry for category {category}")
                except Exception as e:
                    logger.error(f"Error invalidating cache item {key}: {str(e)}")
                    
        except Exception as e:
            logger.error(f"Error invalidating category cache: {str(e)}")
    
    def clear_all(self):
        """Clear all cache entries"""
        try:
            cached_keys = self.redis_client.keys("cache:*")
            if cached_keys:
                self.redis_client.delete(*cached_keys)
                logger.info(f"Cleared {len(cached_keys)} cache entries")
        except Exception as e:
            logger.error(f"Error clearing cache: {str(e)}")