#
# backend/app/agents/retriever_agent.py
#

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
        self.document_service = DocumentService()

    async def process(self, state: AgentState) -> AgentState:
        """
        Processes the user query to retrieve relevant context from the knowledge base.
        """
        logger.info(f"RetrieverAgent: Processing query for category '{state.get('category', 'N/A')}'")
        try:
            # 1. Map the incoming ticket category to a knowledge base category.
            # This determines which specific index/indices to search.
            kb_category = self._get_kb_category(state.get("category"))
            
            if not kb_category:
                logger.warning(f"No knowledge base mapping for category: {state.get('category')}. Skipping retrieval.")
                state["retrieved_context"] = []
                state["current_step"] = WorkflowStep.RESPONSE_GENERATION.value
                return state

            # 2. Use the DocumentService to perform the search.
            # The service handles connecting to the correct index and running the query.
            search_results = await self.document_service.search_documents(
                query=state["processed_query"], # Use the decomposed query for better results
                categories=[kb_category], # Search the single, mapped category
                top_k=10 # Retrieve more results initially for potential re-ranking
            )

            # 3. Process and re-rank the results (optional but good practice).
            # The service already ranks by score, but we can add more logic here.
            reranked_chunks = await self._rerank_chunks(search_results, state["processed_query"])
            
            # 4. Update the state with the top 5 most relevant chunks.
            top_chunks = reranked_chunks[:5]
            state["retrieved_context"] = top_chunks
            state["current_step"] = WorkflowStep.RESPONSE_GENERATION.value
            
            logger.info(f"Retrieved {len(search_results)} chunks, using top {len(top_chunks)} for context.")
            
            return state
            
        except Exception as e:
            logger.error(f"Error in RetrieverAgent: {e}", exc_info=True)
            state["error_message"] = str(e)
            state["requires_escalation"] = True
            state["current_step"] = WorkflowStep.ESCALATION.value
            return state

    def _get_kb_category(self, ticket_category: Optional[str]) -> Optional[str]:
        """
        Maps an incoming ticket category to one of the three main knowledge base categories.
        
        Returns the name of the knowledge base category (e.g., "program_details_documents").
        """
        if not ticket_category:
            return "qa_documents" # Default to qa_documents if no category is provided

        # This mapping connects the application's ticket categories to the
        # high-level knowledge base categories defined in your settings.
        category_mapping = {
            # Program and administrative related -> Program Details
            "Course Query": "program_details_documents",
            "Attendance/Counselling Support": "program_details_documents", 
            "Leave": "program_details_documents",
            "Late Evaluation Submission": "program_details_documents",
            "Missed Evaluation Submission": "program_details_documents",
            "Withdrawal": "program_details_documents",
            
            # Technical and curriculum related -> Curriculum Documents
            "Evaluation Score": "curriculum_Documents",
            "Code Review": "curriculum_Documents",
            "MAC": "curriculum_Documents",
            "Revision": "curriculum_Documents",
            "IA Support": "curriculum_Documents",
            
            # General support, FAQs, troubleshooting -> qa_documents
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
        
        mapped_category = category_mapping.get(ticket_category, "qa_documents")
        logger.info(f"Ticket category '{ticket_category}' mapped to knowledge base: '{mapped_category}'")
        return mapped_category

    async def _rerank_chunks(self, chunks: List[Dict[str, Any]], query: str) -> List[Dict[str, Any]]:
        """
        Re-ranks retrieved chunks. This is a placeholder for a more advanced
        re-ranking model but currently sorts by the initial search score.
        """
        # The DocumentService already returns results sorted by score.
        # A more advanced implementation could use a cross-encoder model here
        # for more accurate relevance ranking.
        # For now, we trust the initial ranking from the vector search.
        chunks.sort(key=lambda x: x.get("score", 0.0), reverse=True)
        return chunks
