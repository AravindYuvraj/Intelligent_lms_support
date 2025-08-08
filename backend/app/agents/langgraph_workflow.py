"""
Enhanced LangGraph Workflow for Intelligent LMS Support System
This module implements a production-ready multi-agent workflow using LangGraph
"""

from typing import Dict, Any, List, Optional, TypedDict, Annotated
from enum import Enum
import asyncio
import logging
from datetime import datetime
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.schema import HumanMessage, AIMessage, SystemMessage
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from backend.app.models import ticket_service, conversation_service, user_service, TicketStatus
from backend.app.core.config import settings
from .cache_service import SemanticCacheService
from .retriever_agent import RetrieverAgent
from .escalation_agent import EscalationAgent

logger = logging.getLogger(__name__)

# Enhanced State Definition with LangGraph annotations
class GraphState(TypedDict):
    """State definition for the LangGraph workflow"""
    # Core ticket information
    ticket_id: str
    user_id: str
    original_query: str
    processed_query: str
    category: str
    subcategory_data: Optional[Dict[str, Any]]
    attachments: Optional[List[str]]
    
    # Workflow tracking
    messages: Annotated[List, add_messages]
    current_step: str
    steps_taken: List[str]
    iteration_count: int
    
    # Agent outputs
    confidence_score: Optional[float]
    response: Optional[str]
    cached_response: Optional[str]
    retrieved_context: Optional[List[Dict[str, Any]]]
    
    # Routing decisions
    admin_type: Optional[str]  # "EC" or "IA"
    requires_escalation: bool
    escalation_reason: Optional[str]
    missing_information: Optional[List[str]]
    
    # Context and metadata
    conversation_history: List[Dict[str, Any]]
    ticket_status: str
    error_message: Optional[str]
    final_status: Optional[str]

class RoutingDecision(BaseModel):
    """Structured output for routing decisions"""
    admin_type: str = Field(description="EC or IA")
    confidence: float = Field(description="Confidence score 0-1")
    reasoning: str = Field(description="Explanation for routing decision")
    missing_info: Optional[List[str]] = Field(default=None, description="Missing information list")
    requires_escalation: bool = Field(default=False)

class ResponseQuality(BaseModel):
    """Response quality assessment"""
    confidence: float = Field(description="Confidence score 0-1")
    completeness: float = Field(description="How complete is the response 0-1")
    requires_human_review: bool
    reasoning: str

class EnhancedLangGraphWorkflow:
    """Production-ready LangGraph workflow with proper error handling and observability"""
    
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
        
        # Initialize agents
        self.retriever_agent = RetrieverAgent()
        self.escalation_agent = EscalationAgent()
        # Build the graph
        self.workflow = self._build_workflow()
    
    def _build_workflow(self) -> StateGraph:
        """Build the LangGraph workflow with proper state transitions"""
        print("Building LangGraph workflow...")
        workflow = StateGraph(GraphState)
        
        # Add nodes
        workflow.add_node("initialize", self.initialize_state)
        workflow.add_node("check_cache", self.check_cache)
        workflow.add_node("route_query", self.route_query)
        workflow.add_node("retrieve_context", self.retrieve_context)
        workflow.add_node("generate_response", self.generate_response)
        workflow.add_node("assess_quality", self.assess_quality)
        workflow.add_node("escalate", self.escalate_to_human)
        workflow.add_node("finalize", self.finalize_ticket)
        
        # Set entry point
        workflow.set_entry_point("initialize")
        
        # Add edges with conditions
        workflow.add_edge("initialize", "check_cache")
        
        workflow.add_conditional_edges(
            "check_cache",
            self.should_use_cache,
            {
                "use_cache": "assess_quality",
                "no_cache": "route_query"
            }
        )
        
        workflow.add_conditional_edges(
            "route_query",
            self.routing_decision,
            {
                "retrieve": "retrieve_context",
                "escalate": "escalate",
                "missing_info": "escalate"
            }
        )
        
        workflow.add_edge("retrieve_context", "generate_response")
        workflow.add_edge("generate_response", "assess_quality")
        
        workflow.add_conditional_edges(
            "assess_quality",
            self.quality_decision,
            {
                "approve": "finalize",
                "escalate": "escalate",
                "retry": "retrieve_context"
            }
        )
        
        workflow.add_edge("escalate", "finalize")
        workflow.add_edge("finalize", END)
        
        return workflow.compile()
    
    async def initialize_state(self, state: GraphState) -> GraphState:
        """Initialize the workflow state with ticket information"""
        try:
            ticket = ticket_service.get_ticket_by_id(state["ticket_id"])
            print(f"Initializing state for ticket {state['ticket_id']}...")
            if not ticket:
                raise ValueError(f"Ticket {state['ticket_id']} not found")
                        
            # The 'category' field is loaded directly from the ticket.
            # This might be a granular category like "IA Support" or "Course Query".
            # The RetrieverAgent is responsible for mapping this to a KB category.
            state["category"] = ticket["category"]
            user = user_service.get_user_by_id(ticket["user_id"])
            conversations = conversation_service.get_ticket_conversations(state["ticket_id"])
            
            # Initialize messages for LangGraph
            messages = [
                HumanMessage(content=ticket["message"])
            ]
            
            state.update({
                "user_id": ticket["user_id"],
                "original_query": ticket["message"],
                "processed_query": ticket["message"],
                "category": ticket["category"],
                "subcategory_data": ticket.get("subcategory_data"),
                "attachments": ticket.get("attachments"),
                "messages": messages,
                "current_step": "initialize",
                "steps_taken": ["initialize"],
                "iteration_count": 0,
                "conversation_history": [
                    {
                        "role": conv["sender_role"],
                        "message": conv["message"],
                        "timestamp": conv["timestamp"].isoformat()
                    } for conv in conversations
                ],
                "ticket_status": ticket["status"],
                "requires_escalation": False
            })
            
            logger.info(f"Initialized state for ticket {state['ticket_id']}")
            return state
            
        except Exception as e:
            logger.error(f"Initialization error: {str(e)}")
            state["error_message"] = str(e)
            state["requires_escalation"] = True
            return state
    
    async def check_cache(self, state: GraphState) -> GraphState:
        """Check semantic cache for similar resolved queries"""
        try:
            cached_result = await self.cache_service.search_similar(
                state["original_query"], 
                threshold=0.85
            )
            print(f"Cache check for ticket {state['ticket_id']}...", cached_result)
            if cached_result:
                state["cached_response"] = cached_result["response"]
                state["confidence_score"] = cached_result.get("confidence", 0.9)
                state["response"] = cached_result["response"]
                state["steps_taken"].append("cache_hit")
                
                # Personalize the cached response
                personalized = await self._personalize_response(
                    cached_result["response"],
                    state["original_query"]
                )
                state["response"] = personalized
                
                logger.info(f"Cache hit for ticket {state['ticket_id']} with confidence {state['confidence_score']}")
            else:
                state["steps_taken"].append("cache_miss")
                logger.info(f"Cache miss for ticket {state['ticket_id']}")
            
            state["current_step"] = "check_cache"
            return state
            
        except Exception as e:
            logger.error(f"Cache check error: {str(e)}")
            state["steps_taken"].append("cache_error")
            return state
    
    async def route_query(self, state: GraphState) -> GraphState:
        """Intelligent query routing with classification and error handling."""
        try:
            parser = PydanticOutputParser(pydantic_object=RoutingDecision)
            
            routing_prompt = ChatPromptTemplate.from_messages([
                ("system", """You are a master query router for Masai School LMS support. Your task is to analyze the student's query and classify it for the correct support team.

                **EC (Experience Champion)** handles:
                - Course logistics: Content, attendance, leaves, evaluations
                - Career support: Placements, industry projects, career guidance
                - Financial matters: ISA/NBFC

                **IA (Instructor Associate)** handles:
                - Technical issues: Interviews, coding assignments, DSA problems
                - Academic support: Code reviews, technical doubts

                Based on the query, decide the admin type, identify any critical missing information, and determine if it requires immediate escalation.
                Respond strictly in the requested JSON format.
                
                {format_instructions}"""),
                ("human", "Query: {query}\nCategory: {category}\nSubcategory Data: {subcategory}")
            ])
            
            chain = routing_prompt | self.llm | parser
            
            routing_decision = await chain.ainvoke({
                "query": state["original_query"], # Use original query for routing context
                "category": state["category"],
                "subcategory": str(state.get("subcategory_data", {})),
                "format_instructions": parser.get_format_instructions()
            })
            
            state["admin_type"] = routing_decision.admin_type
            state["missing_information"] = routing_decision.missing_info
            state["requires_escalation"] = routing_decision.requires_escalation or state.get("requires_escalation", False)
            state["confidence_score"] = routing_decision.confidence
            state["steps_taken"].append(f"routed_to_{routing_decision.admin_type}")
            
            # Decompose query for better retrieval
            decomposed = await self._decompose_query(state["original_query"])
            state["processed_query"] = decomposed
            
            state["current_step"] = "route_query"
            logger.info(f"Routed ticket {state['ticket_id']} to {routing_decision.admin_type}")
            
            return state
            
        except (json.JSONDecodeError, Exception) as e:
            logger.error(f"Routing error (likely JSON parsing failed): {str(e)}")
            state["error_message"] = "Failed to parse routing decision from LLM."
            state["requires_escalation"] = True # Escalate if routing fails
            # Default to a general admin type
            state["admin_type"] = "EC"
            # Use the original query if decomposition fails
            state["processed_query"] = state["original_query"]
            return state
        
    async def retrieve_context(self, state: GraphState) -> GraphState:
        """
        Retrieve context using the RetrieverAgent.
        The agent will internally map the state's category to the correct knowledge base.
        """
        try:
            logger.info(f"Calling RetrieverAgent for ticket {state['ticket_id']} with category: '{state['category']}'")
            
            # The entire state is passed to the agent. The agent's `process` method
            # is designed to work with this state structure. It will look at the
            # `category` field and perform the necessary mapping to find the correct
            # Pinecone index to query.
            retriever_state = await self.retriever_agent.process(state)
            
            # Update the main graph state with the results from the agent
            state["retrieved_context"] = retriever_state.get("retrieved_context", [])
            state["steps_taken"].append("context_retrieved")
            state["current_step"] = "retrieve_context"
            
            logger.info(f"Retrieved {len(state['retrieved_context'])} context chunks for ticket {state['ticket_id']}")
            return state
            
        except Exception as e:
            logger.error(f"Retrieval error: {str(e)}", exc_info=True)
            state["error_message"] = str(e)
            state["retrieved_context"] = []
            state["requires_escalation"] = True # Escalate if retrieval fails
            return state
        
            
    async def generate_response(self, state: GraphState) -> GraphState:
        """Generate a helpful response using a clear persona and retrieved context."""
        try:
            context_text = self._format_context(state.get("retrieved_context", []))
            
            response_prompt = ChatPromptTemplate.from_messages([
                ("system", """You are 'Masai Helper', a friendly and professional AI support agent for Masai School. Your goal is to provide accurate and helpful answers to student queries.

                **Your Persona & Rules:**
                1.  **Be Helpful and Accurate:** Use the provided "Available Context" to construct your answer.
                2.  **NEVER Fabricate:** If the context does not contain the answer, or if no context is provided, you MUST state that you don't have the information. Do not make up dates, policies, or procedures.
                3.  **Acknowledge Missing Context:** If the context seems irrelevant or is missing, explicitly say, "I couldn't find specific information about this in our knowledge base, but I've forwarded your query to our support team."
                4.  **Be Actionable:** Provide clear next steps for the student if possible.

                ---
                Available Context:
                {context}
                ---
                """),
                MessagesPlaceholder(variable_name="messages"),
            ])

            chain = response_prompt | self.llm
            
            response = await chain.ainvoke({
                "context": context_text,
                "messages": state["messages"],
            })
            
            state["response"] = response.content
            state["messages"].append(AIMessage(content=response.content))
            state["steps_taken"].append("response_generated")
            state["current_step"] = "generate_response"
            
            logger.info(f"Generated response for ticket {state['ticket_id']}")
            
            return state
            
        except Exception as e:
            logger.error(f"Response generation error: {str(e)}")
            state["error_message"] = str(e)
            state["response"] = "I apologize, but I encountered an error while generating a response. Your query has been escalated to our human support team who will get back to you shortly."
            state["requires_escalation"] = True
            return state
        
    async def assess_quality(self, state: GraphState) -> GraphState:
        """Assess response quality and determine if escalation is needed"""
        try:
            parser = PydanticOutputParser(pydantic_object=ResponseQuality)
            
            assessment_prompt = ChatPromptTemplate.from_messages([
                ("system", """Assess the quality of this support response.
                Consider: completeness, accuracy, helpfulness, and clarity.
                {format_instructions}"""),
                ("human", """Original Query: {query}
                Generated Response: {response}
                Context Available: {has_context}""")
            ])
            
            chain = assessment_prompt | self.llm | parser
            
            quality = await chain.ainvoke({
                "query": state["original_query"],
                "response": state.get("response", ""),
                "has_context": len(state.get("retrieved_context", [])) > 0,
                "format_instructions": parser.get_format_instructions()
            })
            
            state["confidence_score"] = quality.confidence
            
            if quality.requires_human_review or quality.confidence < 0.85:
                state["requires_escalation"] = True
                state["escalation_reason"] = quality.reasoning
            
            state["steps_taken"].append(f"quality_assessed_{quality.confidence:.2f}")
            state["current_step"] = "assess_quality"
            
            logger.info(f"Quality assessment for ticket {state['ticket_id']}: confidence={quality.confidence:.2f}")
            
            return state
            
        except Exception as e:
            logger.error(f"Quality assessment error: {str(e)}")
            # Default to escalation on error
            state["requires_escalation"] = True
            state["escalation_reason"] = "Quality assessment failed"
            return state
    
    async def escalate_to_human(self, state: GraphState) -> GraphState:
        """Handle escalation to human admin"""
        try:
            escalation_state = await self.escalation_agent.process({
                "ticket_id": state["ticket_id"],
                "user_id": state["user_id"],
                "query": state["original_query"],
                "category": state["category"],
                "admin_type": state.get("admin_type", "EC"),
                "response": state.get("response"),
                "confidence_score": state.get("confidence_score"),
                "missing_information": state.get("missing_information"),
                "error_message": state.get("error_message"),
                "current_step": "escalation",
                "conversation_history": state["conversation_history"],
                "ticket_status": state["ticket_status"],
                "requires_escalation": True
            })
            
            state["final_status"] = TicketStatus.ADMIN_ACTION_REQUIRED.value
            state["steps_taken"].append("escalated_to_human")
            state["current_step"] = "escalate"
            
            logger.info(f"Escalated ticket {state['ticket_id']} to human admin")
            
            return state
            
        except Exception as e:
            logger.error(f"Escalation error: {str(e)}")
            state["error_message"] = str(e)
            return state
    
    async def finalize_ticket(self, state: GraphState) -> GraphState:
        """Finalize ticket processing and update status"""
        try:
            # Determine final status
            if state.get("requires_escalation"):
                final_status = TicketStatus.ADMIN_ACTION_REQUIRED.value
            elif state.get("confidence_score", 0) >= 0.85:
                final_status = TicketStatus.RESOLVED.value
            else:
                final_status = TicketStatus.WIP.value
            
            # Update ticket status
            ticket_service.update_ticket_status(state["ticket_id"], final_status)
            
            # Add conversation entry if we have a response
            if state.get("response"):
                conversation_service.create_conversation(
                    ticket_id=state["ticket_id"],
                    sender_role="agent",
                    message=state["response"],
                    confidence_score=state.get("confidence_score")
                )
            
            state["final_status"] = final_status
            state["steps_taken"].append(f"finalized_{final_status}")
            state["current_step"] = "finalize"
            
            # Log metrics
            logger.info(f"""
            Ticket {state['ticket_id']} processing complete:
            - Final Status: {final_status}
            - Confidence: {state.get('confidence_score', 0):.2f}
            - Steps: {' -> '.join(state['steps_taken'])}
            - Iterations: {state.get('iteration_count', 0)}
            """)
            
            return state
            
        except Exception as e:
            logger.error(f"Finalization error: {str(e)}")
            state["error_message"] = str(e)
            return state
    
    # Helper methods for conditional edges
    def should_use_cache(self, state: GraphState) -> str:
        """Determine if cached response should be used"""
        if state.get("cached_response") and state.get("confidence_score", 0) >= 0.85:
            return "use_cache"
        return "no_cache"
    
    def routing_decision(self, state: GraphState) -> str:
        """Determine next step after routing"""
        if state.get("missing_information"):
            return "missing_info"
        elif state.get("requires_escalation"):
            return "escalate"
        else:
            return "retrieve"
    
    def quality_decision(self, state: GraphState) -> str:
        """Determine action based on quality assessment"""
        if state.get("requires_escalation"):
            return "escalate"
        elif state.get("confidence_score", 0) >= 0.85:
            return "approve"
        elif state.get("iteration_count", 0) < 2:
            state["iteration_count"] = state.get("iteration_count", 0) + 1
            return "retry"
        else:
            return "escalate"
    
    # Helper methods
    async def _personalize_response(self, cached_response: str, query: str) -> str:
        """Personalize a cached response for the current query"""
        prompt = ChatPromptTemplate.from_messages([
            ("system", "Adapt this response to directly address the specific query while keeping the helpful information."),
            ("human", "Query: {query}\nCached Response: {response}\nProvide a personalized version:")
        ])
        
        chain = prompt | self.llm
        result = await chain.ainvoke({"query": query, "response": cached_response})
        return result.content
    
    async def _decompose_query(self, query: str) -> str:
        """Decompose query into concise keywords for better retrieval."""
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert at refining search queries. Your task is to transform a student's conversational query into a concise set of keywords and phrases for a semantic vector search.

            **Instructions:**
            1.  Remove all pleasantries and conversational filler (e.g., 'hello', 'sir', 'please help').
            2.  Focus on the core technical terms, feature names, and the essential problem.
            3.  Keep important entities like 'Campus Connect', 'June', 'May'.
            4.  The output should be a clean, keyword-focused string.

            **Example:**
            - **Original:** "sir i Just got a mail regarding the campus connect in may. But there was another opt in June . So will there be another day in June ??"
            - **Decomposed:** "Campus Connect May June schedule dates"
            """),
            ("human", "Original Query: {query}")
        ])
        
        chain = prompt | self.llm
        result = await chain.ainvoke({"query": query})
        return result.content.strip()
    
    def _format_context(self, context_chunks: List[Dict[str, Any]]) -> str:
        """Format retrieved context for prompt"""
        if not context_chunks:
            return "No specific context available."
        
        formatted = []
        for i, chunk in enumerate(context_chunks[:5], 1):
            formatted.append(f"""
            Source {i}: {chunk.get('filename', 'Unknown')}
            Content: {chunk.get('content', '')[:500]}
            ---""")
        
        return "\n".join(formatted)
    
    async def process_ticket(self, ticket_id: str):
        """Main entry point to process a ticket through the workflow"""
        try:
            initial_state = GraphState(
                ticket_id=ticket_id,
                messages=[],
                steps_taken=[],
                iteration_count=0,
                requires_escalation=False
            )
            
            # Run the workflow
            final_state = await self.workflow.ainvoke(initial_state)
            
            return final_state
            
        except Exception as e:
            logger.error(f"Workflow execution error for ticket {ticket_id}: {str(e)}")
            raise

# Export for use in other modules
workflow_instance = EnhancedLangGraphWorkflow()

async def process_ticket_async(ticket_id: str):
    """Async wrapper for ticket processing"""
    print(f"Processing ticket {ticket_id} through LangGraph workflow...")
    return await workflow_instance.process_ticket(ticket_id)

