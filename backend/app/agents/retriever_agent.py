"""
backend/app/agents/retriever_agent.py
"""

from typing import Dict, Any, List
import logging

# Local application imports
from backend.app.services.document_service import DocumentService
from .state import AgentState, WorkflowStep
from utils import get_kb_category

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
        print(f"RETRIEVER AGENT: Processing ticket {state.get("rewritten_query", "query")}'")
        query = state.get("rewritten_query", state["original_query"])
        
        print(f"RETRIEVER AGENT: Processing query for ticket {ticket_id}")
        print(f"Category: '{category}', Query: '{query[:50]}...'")
        
        try:
            # Check if document service is available
            if not self.document_service:
                raise Exception("DocumentService not initialized")
            
            # 1. Map the incoming ticket category...
            kb_category = get_kb_category(category)
            
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
                top_k=5
            )
            
            high_score_results = [r for r in search_results if r.get("score", 0) >= 0.8]

            if len(high_score_results) < 5 and kb_category != 'qa_documents':
                # Filter out low-score results from main search
                search_results = high_score_results

                # Search in 'qa_documents'
                extra_results = await self.document_service.search_documents(
                    query=query,
                    categories=['qa_documents'],
                    top_k=5
                )

                # Keep only high-score extra results
                extra_results = [r for r in extra_results if r.get("score", 0) >= 0.8]

                # Merge results without duplicates
                search_results.extend(r for r in extra_results if r not in search_results)

            print(f"Raw search results: {len(search_results)} documents found")
            if search_results:
                for i, result in enumerate(search_results[:3]):  # Log first 3 results
                    score = result.get("score", 0)
                    # FIX: Use 'text_snippet' here for the preview
                    content_preview = result.get("text_snippet", "")[:100]
                    filename = result.get("filename", "unknown")
                    print(f"i, result", i, result)
                    print(f"Result {i+1}: score={score:.3f}, file='{filename}', content='{content_preview}...'")

            # 3. Process and re-rank the results
            reranked_chunks = await self._rerank_chunks(search_results, query)
            top_results = reranked_chunks[:5]
                
            # 4. Format the final context, giving priority to Q&A pairs
            final_context = []
            for result in top_results:
                potential_response = result.get("potential_response")
                
                # If a pre-canned answer exists, format it clearly for the LLM
                if potential_response:
                    result['content'] = (
                        f"A relevant Q&A pair was found in the knowledge base.\n"
                        f"Question: {result.get('text_snippet', '')}\n"
                        f"Answer: {potential_response}"
                    )
                else:
                    # For regular documents, the content is just the text snippet
                    result['content'] = result.get('text_snippet', '')
                    
                final_context.append(result)

            # 5. Update the state with the fully prepared context
            state["retrieved_context"] = final_context
            state["current_step"] = WorkflowStep.RESPONSE_GENERATION.value
                    
            context_count = len(final_context)
            if context_count == 0:
                print(f" NO RELEVANT CONTEXT found for ticket {ticket_id}")
            else:
                print(f"SELECTED {context_count} top chunks for context")
                # Log the best chunk
                if final_context:
                    best_chunk = final_context[0]
                    print(f"Best chunk: score={best_chunk.get('score', 0):.3f}, content='{best_chunk.get('content', '')[:100]}...'")

            return state
                
        except Exception as e:
            print(f"RETRIEVER AGENT ERROR for ticket {ticket_id}: ",e)
            state["error_message"] = f"Context retrieval failed: {str(e)}"
            state["requires_escalation"] = True
            state["current_step"] = WorkflowStep.ESCALATION.value
            return state

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
    