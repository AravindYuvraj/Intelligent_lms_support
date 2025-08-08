"""
Enhanced LangGraph Workflow for Intelligent LMS Support System
This module implements a production-ready multi-agent workflow using LangGraph
"""

from typing import Dict, Any, List, Optional, TypedDict, Annotated
from enum import Enum
import asyncio
import logging
import json
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
import traceback
import re

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
            model="gemini-2.0-flash",
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
                "missing_info": "finalize"
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
            print(f"INITIALIZING STATE for ticket {state['ticket_id']}")
            
            ticket = ticket_service.get_ticket_by_id(state["ticket_id"])
            if not ticket:
                raise ValueError(f"Ticket {state['ticket_id']} not found")
                        
            user = user_service.get_user_by_id(ticket["user_id"])
            conversations = conversation_service.get_ticket_conversations(state["ticket_id"])
            
            # Initialize messages for LangGraph
            messages = [
                HumanMessage(content=ticket["message"])
            ]
            
            state.update({
                "user_id": str(ticket["user_id"]),
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
            
            print(f"INITIALIZED STATE: category={state['category']}, query_length={len(state['original_query'])}")
            return state
            
        except Exception as e:
            print(f"INITIALIZATION ERROR")
            state["error_message"] = str(e)
            state["requires_escalation"] = True
            return state
    
    async def check_cache(self, state: GraphState) -> GraphState:
        """Check semantic cache for similar resolved queries"""
        try:
            print(f"CHECKING CACHE for ticket {state['ticket_id']}")
            
            cached_result = await self.cache_service.search_similar(
                state["original_query"], 
                threshold=0.85
            )
            
            if cached_result:
                print(f"CACHE HIT! Similarity: {cached_result.get('similarity', 'N/A')}, Confidence: {cached_result.get('confidence', 'N/A')}")
                
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
                
            else:
                print(f"CACHE MISS for ticket {state['ticket_id']}")
                state["steps_taken"].append("cache_miss")
            
            state["current_step"] = "check_cache"
            return state
            
        except Exception as e:
            print(f"CACHE CHECK ERROR")
            state["steps_taken"].append("cache_error")
            return state
    
    async def route_query(self, state: GraphState) -> GraphState:
        """Intelligent query routing with classification and error handling."""
        try:
            print(f"ROUTING QUERY for ticket {state['ticket_id']}")
            
            routing_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a master query router for Masai School LMS support. Your task is to analyze the student's query and classify it for the correct support team.

**EC (Experience Champion)** handles:
- Course logistics: Content, attendance, leaves, evaluations
- Career support: Placements, industry projects, career guidance
- Financial matters: ISA/NBFC
- Campus events: Campus Connect, orientation programs

**IA (Instructor Associate)** handles:
- Technical issues: Interviews, coding assignments, DSA problems
- Academic support: Code reviews, technical doubts

Respond ONLY with a valid JSON object. Do NOT include triple backticks, markdown, or any explanations. The format should be:

{{
  "admin_type": "EC" or "IA",
  "confidence": float between 0 and 1,
  "reasoning": "Brief explanation",
  "missing_info": null or ["list", "of", "missing", "items"],
  "requires_escalation": true or false
}}
"""),
            ("human", "Query: {query}\nCategory: {category}")
            ])
            
            chain = routing_prompt | self.llm
            
            result = await chain.ainvoke({
                "query": state["original_query"],
                "category": state["category"]
            })
            
            print(f"LLM ROUTING RESPONSE: {result.content[:200]}...")
            
            # Parse JSON response
            try:
                raw = result.content.strip()
                print(f"LLM ROUTING RESPONSE (raw):\n{raw}\n")

                # 1. Strip any Markdown fences (```json …``` or ```)
                if raw.startswith("```"):
                    # split on fences, grab the middle section
                    parts = raw.split("```")
                # e.g. ["", "json\n{...}\n", ""]
                    raw = parts[1] if len(parts) > 1 else parts[0]

                # 2. Extract the first {...} block in case there’s any trailing text
                match = re.search(r"\{.*\}", raw, re.DOTALL)
                json_str = match.group(0) if match else raw
                routing_data = json.loads(json_str)
                print("Parsed routing_data:", routing_data)
            except json.JSONDecodeError as je:
                print(f"JSON PARSE ERROR: {je}")
                # Fallback routing
                routing_data = {
                    "admin_type": "EC",
                    "confidence": 0.5,
                    "reasoning": "JSON parse failed, defaulting to EC",
                    "missing_info": None,
                    "requires_escalation": True
                }
            
            state["admin_type"] = routing_data.get("admin_type", "EC")
            state["missing_information"] = routing_data.get("missing_info")
            state["requires_escalation"] = routing_data.get("requires_escalation", False)
            state["confidence_score"] = routing_data.get("confidence", 0.5)
            state["steps_taken"].append(f"routed_to_{state['admin_type']}")
            
            # Decompose query for better retrieval
            decomposed = await self._decompose_query(state["original_query"])
            state["processed_query"] = decomposed
            
            state["current_step"] = "route_query"
            
            print(f"ROUTING COMPLETE: admin_type={state['admin_type']}, confidence={state['confidence_score']}, escalation_required={state['requires_escalation']}")
            print(f"PROCESSED QUERY: {state['processed_query']}")
            
            return state
            
        except Exception as e:
            print(f"ROUTING ERROR: {e}")
            traceback.print_exc()
            state["error_message"] = f"Routing failed: {str(e)}"
            state["requires_escalation"] = True
            state["admin_type"] = "EC"  # Default fallback
            state["processed_query"] = state["original_query"]
            return state
        
    async def retrieve_context(self, state: GraphState) -> GraphState:
        """Retrieve context using the RetrieverAgent"""
        try:
            print(f"RETRIEVING CONTEXT for ticket {state['ticket_id']} with category: '{state['category']}'")
            
            # Create agent state (convert from GraphState to AgentState format)
            agent_state = {
                "ticket_id": int(state["ticket_id"]) if state["ticket_id"].isdigit() else hash(state["ticket_id"]),
                "user_id": int(state["user_id"]) if state["user_id"].isdigit() else hash(state["user_id"]),
                "query": state["processed_query"],  # Use processed query
                "category": state["category"],
                "subcategory_data": state.get("subcategory_data"),
                "attachments": state.get("attachments"),
                "current_step": "retrieval",
                "confidence_score": state.get("confidence_score"),
                "response": state.get("response"),
                "cached_response": state.get("cached_response"),
                "retrieved_context": state.get("retrieved_context"),
                "admin_type": state.get("admin_type"),
                "requires_escalation": state.get("requires_escalation", False),
                "missing_information": state.get("missing_information"),
                "conversation_history": state.get("conversation_history", []),
                "ticket_status": state.get("ticket_status"),
                "error_message": state.get("error_message")
            }
            
            # Call the retriever agent
            retriever_result = await self.retriever_agent.process(agent_state)
            
            # Update the main graph state with the results
            state["retrieved_context"] = retriever_result.get("retrieved_context", [])
            state["steps_taken"].append("context_retrieved")
            state["current_step"] = "retrieve_context"
            
            # Check if retrieval was successful
            context_count = len(state["retrieved_context"])
            if context_count == 0:
                print(f"NO CONTEXT RETRIEVED for ticket {state['ticket_id']}")
            else:
                print(f"RETRIEVED {context_count} context chunks for ticket {state['ticket_id']}")
                # Log first chunk for debugging
                if state["retrieved_context"]:
                    first_chunk = state["retrieved_context"][0]
                    print(f"FIRST CHUNK: {first_chunk.get('content', '')[:100]}...")
            
            return state
            
        except Exception as e:
            print(f"RETRIEVAL ERROR")
            state["error_message"] = f"Context retrieval failed: {str(e)}"
            state["retrieved_context"] = []
            state["requires_escalation"] = True
            return state
            
    async def generate_response(self, state: GraphState) -> GraphState:
        """Generate a helpful response using retrieved context"""
        try:
            print(f" GENERATING RESPONSE for ticket {state['ticket_id']}")
            
            context_text = self._format_context(state.get("retrieved_context", []))
            print(f"CONTEXT LENGTH: {len(context_text)} characters")
            
            response_prompt = ChatPromptTemplate.from_messages([
                ("system", """You are 'Masai Helper', a friendly and professional AI support agent for Masai School. Your goal is to provide accurate and helpful answers to student queries.
                 
                 Based on the content provided in the "Available Context", construct a response that is:

                **Your Persona & Rules:**
                1. **Be Helpful and Accurate:** Use the provided "Available Context" to construct your answer.
                2. **Be Actionable:** Provide clear next steps for the student if possible.
                3. **Be Friendly:** Use a warm, supportive tone with appropriate emojis.
                4. **Answer the student query directly:** Do not just repeat the query.

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
            
            print(f"GENERATED RESPONSE for ticket {state['ticket_id']}: {len(response.content)} characters")
            print(f"RESPONSE PREVIEW: {response.content[:150]}...")
            
            return state
            
        except Exception as e:
            print(f"RESPONSE GENERATION ERROR")
            state["error_message"] = f"Response generation failed: {str(e)}"
            state["response"] = "I apologize, but I encountered an error while generating a response. Your query has been escalated to our human support team who will get back to you shortly."
            state["requires_escalation"] = True
            return state
        
    async def assess_quality(self, state: GraphState) -> GraphState:
        """Assess response quality and determine if escalation is needed"""
        try:
            print(f"ASSESSING QUALITY for ticket {state['ticket_id']}")
            
            assessment_prompt = ChatPromptTemplate.from_messages([
                ("system", """Assess the quality of this support response. Consider: completeness, accuracy, helpfulness, and clarity.
                
                Respond with ONLY a JSON object in this exact format:
                {{
                    "confidence": 0.95,
                    "completeness": 0.90,
                    "requires_human_review": false,
                    "reasoning": "Brief explanation of the assessment"
                }}"""),
                ("human", """Original Query: {query}
                Generated Response: {response}
                Context Available: {has_context}""")
            ])
            
            chain = assessment_prompt | self.llm
            
            result = await chain.ainvoke({
                "query": state["original_query"],
                "response": state.get("response", ""),
                "has_context": len(state.get("retrieved_context", [])) > 0,
            })
            
            print(f"LLM QUALITY RESPONSE: {result.content[:200]}...")
            
            try:
                raw = result.content.strip()
                print(f"LLM ROUTING RESPONSE (raw):\n{raw}\n")

                # 1. Strip any Markdown fences (```json …``` or ```)
                if raw.startswith("```"):
                # Split on fences, grab the middle section
                    parts = raw.split("```")
                # e.g. ["", "json\n{...}\n", ""]
                raw = parts[1] if len(parts) > 1 else parts[0]

                # 2. Extract the first {...} block in case there’s any trailing text
                match = re.search(r"\{.*\}", raw, re.DOTALL)
                json_str = match.group(0) if match else raw

                quality_data = json.loads(json_str)
                print("Parsed routing_data:", quality_data)

            except json.JSONDecodeError:
                print("Routing JSON parse failed, using fallback")
                quality_data = {
                    "admin_type": "EC",
                    "confidence": 0.5,
                    "reasoning": "JSON parse failed, defaulting to EC",
                    "missing_info": None,
                    "requires_escalation": True
                }
            
            state["confidence_score"] = quality_data.get("confidence", 0.5)
            
            if quality_data.get("requires_human_review", False) or state["confidence_score"] < 0.85:
                state["requires_escalation"] = True
                state["escalation_reason"] = quality_data.get("reasoning", "Low confidence score")
            
            state["steps_taken"].append(f"quality_assessed_{state['confidence_score']:.2f}")
            state["current_step"] = "assess_quality"
            
            print(f"QUALITY ASSESSMENT COMPLETE: confidence={state['confidence_score']:.2f}, escalation_required={state['requires_escalation']}")
            
            return state
            
        except Exception as e:
            print(f"QUALITY ASSESSMENT ERROR")
            # Default to escalation on error
            state["requires_escalation"] = True
            state["escalation_reason"] = f"Quality assessment failed: {str(e)}"
            state["confidence_score"] = 0.3
            return state
    
    async def escalate_to_human(self, state: GraphState) -> GraphState:
        """Handle escalation to human admin"""
        try:
            print(f"ESCALATING to human for ticket {state['ticket_id']}")
            
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
            
            print(f"ESCALATED ticket {state['ticket_id']} to human admin ({state.get('admin_type', 'EC')})")
            
            return state
            
        except Exception as e:
            print(f"ESCALATION ERROR")
            state["error_message"] = f"Escalation failed: {str(e)}"
            return state
    
    async def finalize_ticket(self, state: GraphState) -> GraphState:
        """Finalize ticket processing and update status"""
        try:
            print(f"FINALIZING ticket {state['ticket_id']}")
            
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
            if state.get("response") and not state.get("requires_escalation"):
                conversation_service.create_conversation(
                    ticket_id=state["ticket_id"],
                    sender_role="agent",
                    message=state["response"],
                    confidence_score=state.get("confidence_score")
                )
                
                # Cache the response if it's high quality
                if state.get("confidence_score", 0) >= 0.85:
                    try:
                        await self.cache_service.store_response(
                            query=state["original_query"],
                            response=state["response"],
                            confidence=state["confidence_score"],
                            category=state["category"]
                        )
                        print(f"CACHED response for future use")
                    except Exception as cache_error:
                        print(f"Cache storage failed: {cache_error}")
            
            state["final_status"] = final_status
            state["steps_taken"].append(f"finalized_{final_status}")
            state["current_step"] = "finalize"
            
            # Log comprehensive metrics
            print(f"""TICKET {state['ticket_id']} PROCESSING COMPLETE:
            - Final Status: {final_status}
            - Confidence: {state.get('confidence_score', 0):.2f}
            - Steps: {' -> '.join(state['steps_taken'])}
            - Iterations: {state.get('iteration_count', 0)}
            - Admin Type: {state.get('admin_type', 'N/A')}
            - Context Chunks: {len(state.get('retrieved_context') or [])}
            """)
            
            return state
            
        except Exception as e:
            print(f"FINALIZATION ERROR")
            state["error_message"] = f"Finalization failed: {str(e)}"
            return state
    
    # Helper methods for conditional edges
    def should_use_cache(self, state: GraphState) -> str:
        """Determine if cached response should be used"""
        if state.get("cached_response") and state.get("confidence_score", 0) >= 0.85:
            print(f"USING CACHED RESPONSE for ticket {state['ticket_id']}")
            return "use_cache"
        print(f"⏭ BYPASSING CACHE, proceeding to routing")
        return "no_cache"
    
    def routing_decision(self, state: GraphState) -> str:
        """Determine next step after routing"""
        if state.get("missing_information"):
            print(f"MISSING INFO detected")
            return "missing_info"
        elif state.get("requires_escalation"):
            print(f"ESCALATION REQUIRED from routing")
            return "escalate"
        else:
            print(f"PROCEEDING TO RETRIEVAL")
            return "retrieve"
    
    def quality_decision(self, state: GraphState) -> str:
        """Determine action based on quality assessment"""
        if state.get("requires_escalation"):
            print(f"ESCALATION REQUIRED from quality check")
            return "escalate"
        elif state.get("confidence_score", 0) >= 0.85:
            print(f"HIGH QUALITY RESPONSE, approving")
            return "approve"
        elif state.get("iteration_count", 0) < 2:
            state["iteration_count"] = state.get("iteration_count", 0) + 1
            print(f"RETRYING retrieval, iteration {state['iteration_count']}")
            return "retry"
        else:
            print(f"MAX RETRIES REACHED, escalating")
            return "escalate"
    
    # Helper methods
    async def _personalize_response(self, cached_response: str, query: str) -> str:
        """Personalize a cached response for the current query"""
        try:
            prompt = ChatPromptTemplate.from_messages([
                ("system", "Adapt this response to directly address the specific query while keeping the helpful information."),
                ("human", "Query: {query}\nCached Response: {response}\nProvide a personalized version:")
            ])
            
            chain = prompt | self.llm
            result = await chain.ainvoke({"query": query, "response": cached_response})
            return result.content
        except Exception as e:
            print(f"Personalization failed: {e}, using original cached response")
            return cached_response
    
    async def _decompose_query(self, query: str) -> str:
        """Decompose query into concise keywords for better retrieval."""
        try:
            prompt = ChatPromptTemplate.from_messages([
                ("system", """You are an expert at refining search queries. Transform a student's conversational query into concise keywords for semantic vector search.

                **Instructions:**
                1. Remove pleasantries and filler (e.g., 'hello', 'sir', 'please help')
                2. Focus on core technical terms, feature names, and essential problems
                3. Keep important entities like 'Campus Connect', 'June', 'May'
                4. Output should be clean, keyword-focused

                **Example:**
                - Original: "sir i Just got a mail regarding the campus connect in may. But there was another opt in June . So will there be another day in June ??"
                - Decomposed: "Campus Connect May June schedule dates"
                """),
                ("human", "Original Query: {query}")
            ])
            
            chain = prompt | self.llm
            result = await chain.ainvoke({"query": query})
            decomposed = result.content.strip()
            
            print(f"QUERY DECOMPOSITION: '{query}' -> '{decomposed}'")
            return decomposed
            
        except Exception as e:
            print(f"Query decomposition failed: {e}, using original query")
            return query
    
    def _format_context(self, context_chunks: List[Dict[str, Any]]) -> str:
        """Format retrieved context for prompt"""
        if not context_chunks:
            return "No specific context available."
        
        formatted = []
        for i, chunk in enumerate(context_chunks[:5], 1):
            content = chunk.get('content', '')[:500]
            filename = chunk.get('filename', 'Unknown')
            formatted.append(f"Source {i}: {filename}\nContent: {content}\n---")
        
        result = "\n".join(formatted)
        print(f"FORMATTED CONTEXT: {len(result)} characters from {len(context_chunks)} chunks")
        return result
    
    async def process_ticket(self, ticket_id: str):
        """Main entry point to process a ticket through the workflow"""
        try:
            print(f"STARTING WORKFLOW for ticket {ticket_id}")
            
            initial_state = GraphState(
                ticket_id=ticket_id,
                user_id="",  # Will be populated in initialize
                original_query="",  # Will be populated in initialize
                processed_query="",  # Will be populated in routing
                category="",  # Will be populated in initialize
                subcategory_data=None,
                attachments=None,
                messages=[],
                current_step="",
                steps_taken=[],
                iteration_count=0,
                confidence_score=None,
                response=None,
                cached_response=None,
                retrieved_context=None,
                admin_type=None,
                requires_escalation=False,
                escalation_reason=None,
                missing_information=None,
                conversation_history=[],
                ticket_status="",
                error_message=None,
                final_status=None
            )
            
            # Run the workflow
            print(f"EXECUTING LANGGRAPH WORKFLOW for ticket {ticket_id}")
            final_state = await self.workflow.ainvoke(initial_state)
            
            print(f"WORKFLOW COMPLETED for ticket {ticket_id}")
            return final_state
            
        except Exception as e:
            print(f"WORKFLOW EXECUTION ERROR for ticket {ticket_id}")
            raise

# Export for use in other modules
workflow_instance = EnhancedLangGraphWorkflow()

async def process_ticket_async(ticket_id: str):
    """Async wrapper for ticket processing"""
    print(f"PROCESSING ticket {ticket_id} through LangGraph workflow...")
    return await workflow_instance.process_ticket(ticket_id)