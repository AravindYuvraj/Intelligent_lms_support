"""
backend/app/agents/retriever_agent.py
"""

from typing import Dict, Any, List, Optional
import logging

# Local application imports
from backend.app.services.document_service import DocumentService
from .state import AgentState, WorkflowStep

logger = logging.getLogger(__name__)

class RetrieverAgent:
    """
    An agent responsible for retrieving relevant context from the knowledge base
    by leveraging the multi-index capabilities of the DocumentService.
    """
    def __init__(self):
        # The agent now uses the DocumentService as its single point of contact
        # for all data retrieval, abstracting away direct database connections.
        try:
            self.document_service = DocumentService()
            print("RetrieverAgent initialized successfully")
        except Exception as e:
            print(f"Failed to initialize DocumentService: {e}")
            self.document_service = None

    async def process(self, state: AgentState) -> AgentState:
        """
        Processes the user query to retrieve relevant context from the knowledge base.
        """
        ticket_id = state.get("ticket_id", "unknown")
        category = state.get("category", "N/A")
        query = state.get("query", "")
        
        print(f"RETRIEVER AGENT: Processing query for ticket {ticket_id}")
        print(f"Category: '{category}', Query: '{query[:50]}...'")
        
        try:
            # Check if document service is available
            if not self.document_service:
                raise Exception("DocumentService not initialized")
            
            # 1. Map the incoming ticket category to a knowledge base category.
            kb_category = self._get_kb_category(category)
            
            if not kb_category:
                print(f"No knowledge base mapping for category: {category}. Skipping retrieval.")
                state["retrieved_context"] = []
                state["current_step"] = WorkflowStep.RESPONSE_GENERATION.value
                return state

            print(f"Mapped to KB category: '{kb_category}'")

            # 2. Use the DocumentService to perform the search.
            print(f"Searching with query: '{query}'")
            search_results = await self.document_service.search_documents(
                query=query,
                categories=[kb_category],
                top_k=10
            )

            print(f"Raw search results: {len(search_results)} documents found")
            if search_results:
                for i, result in enumerate(search_results[:3]):  # Log first 3 results
                    score = result.get("score", 0)
                    content_preview = result.get("content", "")[:100]
                    filename = result.get("filename", "unknown")
                    print(f"Result {i+1}: score={score:.3f}, file='{filename}', content='{content_preview}...'")

            # 3. Process and re-rank the results
            reranked_chunks = await self._rerank_chunks(search_results, query)
            
            # 4. Update the state with the top 5 most relevant chunks.
            top_chunks = reranked_chunks[:5]
            state["retrieved_context"] = top_chunks
            state["current_step"] = WorkflowStep.RESPONSE_GENERATION.value
            
            context_count = len(top_chunks)
            if context_count == 0:
                print(f" NO RELEVANT CONTEXT found for ticket {ticket_id}")
            else:
                print(f"SELECTED {context_count} top chunks for context")
                # Log the best chunk
                if top_chunks:
                    best_chunk = top_chunks[0]
                    print(f"Best chunk: score={best_chunk.get('score', 0):.3f}, content='{best_chunk.get('content', '')[:100]}...'")
            
            return state
            
        except Exception as e:
            print(f"RETRIEVER AGENT ERROR for ticket {ticket_id}: {e}", exc_info=True)
            state["error_message"] = f"Context retrieval failed: {str(e)}"
            state["requires_escalation"] = True
            state["current_step"] = WorkflowStep.ESCALATION.value
            return state

    def _get_kb_category(self, ticket_category: Optional[str]) -> Optional[str]:
        """
        Maps an incoming ticket category to one of the three main knowledge base categories.
        
        Returns the name of the knowledge base category (e.g., "program_details_documents").
        """
        if not ticket_category:
            print("No category provided, defaulting to qa_documents")
            return "qa_documents"

        # Enhanced mapping with more detailed logging
        category_mapping = {
            # Program and administrative related -> Program Details
            "Course Query": "program_details_documents",
            "Attendance/Counselling Support": "program_details_documents", 
            "Leave": "program_details_documents",
            "Late Evaluation Submission": "program_details_documents",
            "Missed Evaluation Submission": "program_details_documents",
            "Withdrawal": "program_details_documents",
            
            # Technical and curriculum related -> Curriculum Documents
            "Evaluation Score": "curriculum_documents",
            "MAC": "curriculum_documents",
            "Revision": "curriculum_documents",
            "IA Support": "curriculum_documents",
            
            # General support, FAQs, troubleshooting -> qa_documents
            "Product Support": "qa_documents",
            "NBFC/ISA": "qa_documents",
            "Feedback": "qa_documents",
            "Referral": "qa_documents",
            "Personal Query": "qa_documents",
            "Code Review": "qa_documents",
            "Placement Support - Placements": "qa_documents",
            "Offer Stage- Placements": "qa_documents", 
            "ISA/EMI/NBFC/Glide Related - Placements": "qa_documents",
            "Session Support - Placement": "qa_documents",
        }
        
        mapped_category = category_mapping.get(ticket_category)
        
        if mapped_category:
            print(f"CATEGORY MAPPING: '{ticket_category}' -> '{mapped_category}'")
        else:
            print(f" UNMAPPED CATEGORY: '{ticket_category}', defaulting to 'qa_documents'")
            mapped_category = "qa_documents"
        
        return mapped_category

    async def _rerank_chunks(self, chunks: List[Dict[str, Any]], query: str) -> List[Dict[str, Any]]:
        """
        Re-ranks retrieved chunks based on relevance score.
        """
        if not chunks:
            print("No chunks to rerank")
            return chunks
        
        print(f"RERANKING {len(chunks)} chunks")
        
        # Sort by score (assuming higher scores are better)
        try:
            sorted_chunks = sorted(chunks, key=lambda x: x.get("score", 0.0), reverse=True)
            
            # Log reranking results
            if len(sorted_chunks) > 0:
                best_score = sorted_chunks[0].get("score", 0)
                worst_score = sorted_chunks[-1].get("score", 0)
                print(f"Reranked: best_score={best_score:.3f}, worst_score={worst_score:.3f}")
            
            return sorted_chunks
            
        except Exception as e:
            print(f"Reranking failed: {e}, returning original order")
            return chunks
    
    def get_supported_categories(self) -> List[str]:
        """Return list of supported knowledge base categories"""
        return [
            "program_details_documents",
            "curriculum_documents", 
            "qa_documents"
        ]
    
    def get_category_mapping(self) -> Dict[str, str]:
        """Return the complete category mapping for debugging"""
        return {
            "Course Query": "program_details_documents",
            "Attendance/Counselling Support": "program_details_documents", 
            "Leave": "program_details_documents",
            "Late Evaluation Submission": "program_details_documents",
            "Missed Evaluation Submission": "program_details_documents",
            "Withdrawal": "program_details_documents",
            "Evaluation Score": "curriculum_documents",
            "Code Review": "curriculum_documents",
            "MAC": "curriculum_documents",
            "Revision": "curriculum_documents",
            "IA Support": "curriculum_documents",
            "Product Support": "qa_documents",
            "NBFC/ISA": "qa_documents",
            "Feedback": "qa_documents",
            "Referral": "qa_documents",
            "Personal Query": "qa_documents",
            "Placement Support - Placements": "qa_documents",
            "Offer Stage- Placements": "qa_documents", 
            "ISA/EMI/NBFC/Glide Related - Placements": "qa_documents",
            "Session Support - Placement": "qa_documents",
        }