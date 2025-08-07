from typing import Dict, Any
import json
import numpy as np
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from backend.app.db.base import get_redis
from backend.app.models import ticket_service, conversation_service, user_service, TicketStatus, UserRole
from backend.app.core.config import settings
from .state import AgentState, WorkflowStep
from .cache_service import SemanticCacheService
import logging

logger = logging.getLogger(__name__)

class RoutingAgent:
    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash-exp",
            google_api_key=settings.GOOGLE_API_KEY,
            temperature=0.1
        )
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-001",
            google_api_key=settings.GOOGLE_API_KEY
        )
        self.cache_service = SemanticCacheService()
    
    async def process(self, state: AgentState) -> AgentState:
        """Main routing logic - check cache, triage, and decompose query"""
        try:
            # Step 1: Check semantic cache
            cached_result = await self.check_cache(state["query"])
            if cached_result:
                state["cached_response"] = cached_result["response"]
                state["confidence_score"] = cached_result.get("confidence", 0.9)
                state["current_step"] = WorkflowStep.COMPLETION.value
                return state
            
            # Step 2: Triage admin type and check for missing information
            triage_result = await self.triage_query(state)
            state.update(triage_result)
            
            # Step 3: If missing information, mark for escalation
            if state.get("missing_information"):
                state["requires_escalation"] = True
                state["current_step"] = WorkflowStep.ESCALATION.value
                return state
            
            # Step 4: Decompose query for retrieval
            decomposed_query = await self.decompose_query(state["query"], state["category"])
            state["query"] = decomposed_query
            state["current_step"] = WorkflowStep.RETRIEVAL.value
            
            return state
            
        except Exception as e:
            logger.error(f"Error in routing agent: {str(e)}")
            state["error_message"] = str(e)
            state["requires_escalation"] = True
            state["current_step"] = WorkflowStep.ESCALATION.value
            return state
    
    async def check_cache(self, query: str) -> Dict[str, Any]:
        """Check Redis semantic cache for similar queries"""
        try:
            return await self.cache_service.search_similar(query, threshold=0.85)
        except Exception as e:
            logger.error(f"Cache check error: {str(e)}")
            return None
    
    async def triage_query(self, state: AgentState) -> Dict[str, Any]:
        """Determine admin type (EC/IA) and check for missing information"""
        
        triage_prompt = f"""
        Analyze this support ticket and determine:
        
        1. Admin Type: Should this be handled by EC (Experience Champion) or IA (Instructor Associate)?
        2. Missing Information: Is there any critical information missing that would prevent resolution?
        
        Ticket Details:
        Category: {state['category']}
        Query: {state['query']}
        Subcategory Data: {json.dumps(state.get('subcategory_data', {}), indent=2)}
        
        Guidelines:
        - EC handles: Course content, attendance, leaves, evaluations, Placements, industry projects, career guidance, ISA/NBFC
        - IA handles: Technical interviews, general academic queries, coding assignments, DSA, and technical doubts
        - If query is clear and complete, return admin_type as "EC" or "IA" based on category
        - If query requires escalation, set requires_escalation to true
        - If query is unclear or missing critical details, list what information is needed
        
        Respond in JSON format:
        {{
            "admin_type": "EC" or "IA",
            "reasoning": "explanation",
            "missing_information": ["list of missing details"] or null,
            "requires_escalation": true/false
        }}
        """
        
        try:
            response = await self.llm.ainvoke(triage_prompt)
            result = json.loads(response.content)
            
            return {
                "admin_type": result["admin_type"],
                "missing_information": result["missing_information"],
                "requires_escalation": result["requires_escalation"]
            }
        except Exception as e:
            logger.error(f"Triage error: {str(e)}")
            return {
                "admin_type": "EC",  # Default
                "missing_information": None,
                "requires_escalation": True
            }
    
    async def decompose_query(self, query: str, category: str) -> str:
        """Break down the query into searchable components"""
        
        decompose_prompt = f"""
        Break down this support query into key searchable terms and concepts for knowledge base retrieval.
        
        Original Query: {query}
        Category: {category}
        
        Extract and reformulate the query to focus on:
        1. Core problem/issue
        2. Relevant product/system names
        3. Specific error messages or symptoms
        4. Action being attempted
        
        Return a clear, focused search query that would find relevant documentation:
        """
        
        try:
            response = await self.llm.ainvoke(decompose_prompt)
            return response.content.strip()
        except Exception as e:
            logger.error(f"Query decomposition error: {str(e)}")
            return query  # Return original if decomposition fails


def process_ticket(ticket_id: int):
    """Background task to process ticket through LangGraph workflow"""
    # This will be the main entry point that orchestrates the entire workflow
    from backend.app.agents.workflow import TicketWorkflow
    
    workflow = TicketWorkflow()
    workflow.run(ticket_id)