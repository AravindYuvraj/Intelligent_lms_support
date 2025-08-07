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
        # Map ticket categories to our 3 EXACT knowledge base collection names:
        # 1. qa_documents - FAQs, common questions, troubleshooting
        # 2. program_details_documents - course info, schedules, policies, timelines
        # 3. curriculum_documents - technical content, assignments, evaluations
        
        category_mapping = {
            # Program and administrative related → program_details_documents
            "Course Query": "program_details_documents",
            "Attendance/Counselling Support": "program_details_documents", 
            "Leave": "program_details_documents",  # Leave policies are program details
            "Late Evaluation Submission": "program_details_documents",  # Submission policies
            "Missed Evaluation Submission": "program_details_documents",  # Evaluation policies
            "Withdrawal": "program_details_documents",  # Withdrawal policies
            
            # Technical and curriculum related → curriculum_documents
            "Evaluation Score": "curriculum_documents",
            "Code Review": "curriculum_documents",
            "MAC": "curriculum_documents",  # Masai Additional Curriculum
            "Revision": "curriculum_documents",  # Course content revision
            "IA Support": "curriculum_documents",  # Technical support from IA
            
            # General support, FAQs, troubleshooting → qa_documents
            "Product Support": "qa_documents",
            "NBFC/ISA": "qa_documents",  # Financial FAQs
            "Feedback": "qa_documents",
            "Referral": "qa_documents",
            "Personal Query": "qa_documents",
            
            # Placement related → qa_documents (mostly FAQs about placement)
            "Placement Support - Placements": "qa_documents",
            "Offer Stage- Placements": "qa_documents", 
            "ISA/EMI/NBFC/Glide Related - Placements": "qa_documents",
            "Session Support - Placement": "qa_documents",
        }
        
        # Get the mapped category, default to qa_documents for unknown categories
        mapped_category = category_mapping.get(category, "qa_documents")
        
        # Log the mapping for debugging
        import logging
        logger = logging.getLogger(__name__)
        logger.info(f"Ticket category '{category}' mapped to knowledge base: '{mapped_category}'")
        
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