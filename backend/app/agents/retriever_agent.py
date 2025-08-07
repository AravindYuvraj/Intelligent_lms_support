from typing import Dict, Any, List
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from pinecone import Pinecone
from backend.app.core.config import settings
from .state import AgentState, WorkflowStep
import logging

logger = logging.getLogger(__name__)

class RetrieverAgent:
    def __init__(self):
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-001",
            google_api_key=settings.GOOGLE_API_KEY
        )
        
        if settings.PINECONE_API_KEY:
            self.pinecone = Pinecone(api_key=settings.PINECONE_API_KEY)
            try:
                self.index = self.pinecone.Index(settings.PINECONE_INDEX_NAME)
            except Exception as e:
                logger.error(f"Failed to connect to Pinecone index: {str(e)}")
                self.index = None
        else:
            self.index = None
    
    async def process(self, state: AgentState) -> AgentState:
        """Retrieve relevant context from knowledge base"""
        try:
            if not self.index:
                logger.warning("Pinecone index not available")
                state["retrieved_context"] = []
                state["current_step"] = WorkflowStep.RESPONSE_GENERATION.value
                return state
            
            # Generate embedding for the query
            query_embedding = await self.embeddings.aembed_query(state["query"])
            
            # Search in Pinecone with category filter
            category_filter = self._get_category_filter(state["category"])
            
            search_results = self.index.query(
                vector=query_embedding,
                filter=category_filter,
                top_k=10,
                include_metadata=True,
                include_values=False
            )
            
            # Extract and re-rank results
            retrieved_chunks = []
            for match in search_results.matches:
                chunk_data = {
                    "content": match.metadata.get("content", ""),
                    "score": match.score,
                    "filename": match.metadata.get("filename", ""),
                    "category": match.metadata.get("category", ""),
                    "doc_id": match.metadata.get("doc_id", "")
                }
                retrieved_chunks.append(chunk_data)
            
            # Re-rank based on relevance and any previous ratings
            reranked_chunks = await self._rerank_chunks(retrieved_chunks, state["query"])
            
            state["retrieved_context"] = reranked_chunks[:5]  # Top 5 chunks
            state["current_step"] = WorkflowStep.RESPONSE_GENERATION.value
            
            logger.info(f"Retrieved {len(retrieved_chunks)} chunks, using top {len(state['retrieved_context'])}")
            
            return state
            
        except Exception as e:
            logger.error(f"Error in retriever agent: {str(e)}")
            state["error_message"] = str(e)
            state["requires_escalation"] = True
            state["current_step"] = WorkflowStep.ESCALATION.value
            return state
    
    def _get_category_filter(self, category: str) -> Dict[str, Any]:
        """Get Pinecone filter based on ticket category - map to 3 knowledge base categories"""
        # Map ticket categories to our 3 knowledge base categories:
        # 1. Program Details - course info, schedules, structure
        # 2. Q&A - FAQs, common questions, troubleshooting  
        # 3. Curriculum Documents - technical content, assignments, evaluations
        
        category_mapping = {
            # Course and program related
            "Course Query": "Program Details",
            "Attendance/Counselling Support": "Program Details", 
            "Revision": "Program Details",
            "Late Evaluation Submission": "Program Details",
            "Missed Evaluation Submission": "Program Details",
            
            # Technical and curriculum related
            "Evaluation Score": "Curriculum Documents",
            "Code Review": "Curriculum Documents",
            "MAC": "Curriculum Documents",
            "Session Support - Placement": "Curriculum Documents",
            
            # Support, troubleshooting, FAQs
            "Product Support": "Q&A",
            "IA Support": "Q&A", 
            "NBFC/ISA": "Q&A",
            "Feedback": "Q&A",
            "Withdrawal": "Q&A",
            
            # Placement related - could go to Q&A for general questions
            "Placement Support - Placements": "Q&A",
            "Offer Stage- Placements": "Q&A", 
            "ISA/EMI/NBFC/Glide Related - Placements": "Q&A",
            
            # Admin/procedural
            "Leave": "Q&A",
            "Referral": "Q&A",
            "Personal Query": "Q&A"
        }
        
        mapped_category = category_mapping.get(category, "Q&A")  # Default to Q&A
        
        return {"category": {"$eq": mapped_category}}
    
    async def _rerank_chunks(self, chunks: List[Dict[str, Any]], query: str) -> List[Dict[str, Any]]:
        """Re-rank retrieved chunks based on relevance and quality"""
        try:
            # Simple re-ranking based on score and content length
            for chunk in chunks:
                # Boost score for longer, more detailed content
                content_length_bonus = min(len(chunk["content"]) / 1000, 0.1)
                chunk["final_score"] = chunk["score"] + content_length_bonus
            
            # Sort by final score
            chunks.sort(key=lambda x: x["final_score"], reverse=True)
            
            return chunks
            
        except Exception as e:
            logger.error(f"Reranking error: {str(e)}")
            return chunks  # Return original order if reranking fails