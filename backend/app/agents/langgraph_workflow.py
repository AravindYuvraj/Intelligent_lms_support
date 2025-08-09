"""
Enhanced LangGraph Workflow for Intelligent LMS Support System
This module implements a production-ready multi-agent workflow using LangGraph
"""

from typing import Dict, Any, List, Optional, TypedDict, Annotated
import asyncio
import logging
import json
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import HumanMessage, AIMessage, SystemMessage
from langchain.prompts import ChatPromptTemplate,MessagesPlaceholder
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
    category: str
    
    # Workflow tracking
    messages: Annotated[List, add_messages]
    steps_taken: List[str]
    
    # Agent inputs and outputs
    context: str # Unified context from cache or retriever
    agent_decision: Dict[str, Any] # Structured decision from the LLM
    
    # Context and metadata
    conversation_history: List[Dict[str, Any]]
    final_status: Optional[str]
    error_message: Optional[str]

class AgentDecision(BaseModel):
    """Structured output for the agent's decision-making process."""
    decision: str = Field(description="The final decision. Must be one of: 'respond', 'request_info', 'escalate'.")
    response: Optional[str] = Field(default=None, description="The generated response for the student if the decision is 'respond'.")
    missing_info: Optional[List[str]] = Field(default=None, description="A list of specific information required from the student if the decision is 'request_info'.")
    escalation_reason: Optional[str] = Field(default=None, description="A brief reason for escalation if the decision is 'escalate'.")
    admin_type: str = Field(description="The team responsible for the query. Must be 'EC' or 'IA'.")
    confidence: float = Field(description="Confidence score (0.0 to 1.0) in the decision.")

class EnhancedLangGraphWorkflow:
    """Production-ready LangGraph workflow with a central decision-making agent."""
    
    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            google_api_key=settings.GOOGLE_API_KEY,
            temperature=0.1,
            # Instruct the model to return JSON
            model_kwargs={"response_mime_type": "application/json"}
        )
        self.cache_service = SemanticCacheService()
        self.retriever_agent = RetrieverAgent()
        self.escalation_agent = EscalationAgent()
        self.workflow = self._build_workflow()
        
    async def _find_available_admin(self, admin_type: str) -> Dict[str, Any]:
        """
        Finds an available admin.
        In a real system, this would check for load, online status, and specialty.
        For now, it returns the first available admin.
        """
        try:
            # This logic can be expanded to filter by EC/IA roles if they are stored in the user model
            admins = user_service.get_admins(admin_type=admin_type)
            print(f"available admins {admins}.")
            return admins[0] if admins else None
        except Exception as e:
            logger.error(f"Error finding admin: {e}")
            return None

    def _build_workflow(self) -> StateGraph:
        """Builds the simplified, more powerful LangGraph workflow."""
        print("Building LangGraph workflow...")
        workflow = StateGraph(GraphState)
        
        # Define the nodes
        workflow.add_node("initialize", self.initialize_state)
        workflow.add_node("check_cache", self.check_cache)
        workflow.add_node("retrieve_context", self.retrieve_context)
        workflow.add_node("generate_and_decide", self.generate_and_decide)
        workflow.add_node("finalize_and_act", self.finalize_and_act)
        
        # Define the graph structure
        workflow.set_entry_point("initialize")
        workflow.add_edge("initialize", "check_cache")
        
        workflow.add_conditional_edges(
            "check_cache",
            lambda state: "retrieve_context" if state.get("context") is None else "generate_and_decide",
            {
                "retrieve_context": "retrieve_context",
                "generate_and_decide": "generate_and_decide"
            }
        )
        
        workflow.add_edge("retrieve_context", "generate_and_decide")
        workflow.add_edge("generate_and_decide", "finalize_and_act")
        workflow.add_edge("finalize_and_act", END)
        
        return workflow.compile()

    async def initialize_state(self, state: GraphState) -> GraphState:
        """Initialize the workflow state with ticket information."""
        print(f"INITIALIZING STATE for ticket {state['ticket_id']}")
        try:
            ticket = ticket_service.get_ticket_by_id(state["ticket_id"])
            if not ticket:
                raise ValueError(f"Ticket {state['ticket_id']} not found")

            conversations = conversation_service.get_ticket_conversations(state["ticket_id"])
            
            messages = []
            for conv in conversations:
                if conv["sender_role"] == "student":
                    messages.append(HumanMessage(content=conv["message"]))
                else:
                    # Assuming 'agent' or 'support' roles are the AI
                    messages.append(AIMessage(content=conv["message"]))
            
            state.update({
                "user_id": str(ticket["user_id"]),
                "original_query": conversations[-1]['message'], # The latest message is the current query
                "category": ticket["category"],
                "messages": [HumanMessage(content=conversations[-1]['message'])],
                "steps_taken": ["initialize"]
            })
            print(f"INITIALIZED STATE: category={state['category']}, query='{state['original_query'][:50]}...'")
            return state
        except Exception as e:
            print(f"INITIALIZATION ERROR: {e}")
            state["error_message"] = str(e)
            return state

    async def check_cache(self, state: GraphState) -> GraphState:
        """Check semantic cache for similar resolved queries."""
        print(f"CHECKING CACHE for ticket {state['ticket_id']}")
        cached_result = await self.cache_service.search_similar(
            state["original_query"], threshold=0.85
        )
        
        if cached_result:
            print(f"CACHE HIT! Similarity: {cached_result.get('similarity', 'N/A')}")
            state["context"] = f"A similar query was resolved in the past.\nCached Response: {cached_result['response']}"
            state["steps_taken"].append("cache_hit")
        else:
            print(f"CACHE MISS for ticket {state['ticket_id']}")
            state["context"] = None # Explicitly set to None for the conditional edge
            state["steps_taken"].append("cache_miss")
        return state

    async def retrieve_context(self, state: GraphState) -> GraphState:
        """Retrieve context using the RetrieverAgent on cache miss."""
        print(f"RETRIEVING CONTEXT for ticket {state['ticket_id']}")
        try:
            retriever_result = await self.retriever_agent.process({
                "query": state["original_query"],
                "category": state["category"]
            })
            retrieved_docs = retriever_result.get("retrieved_context", [])
            
            if not retrieved_docs:
                state["context"] = "No relevant documents were found in the knowledge base."
            else:
                formatted_context = "\n---\n".join([
                    f"Source: {doc.get('filename', 'N/A')}\nContent: {doc.get('content', '')}"
                    for doc in retrieved_docs
                ])
                state["context"] = formatted_context
            
            state["steps_taken"].append("context_retrieved")
            return state
        except Exception as e:
            print(f"RETRIEVAL ERROR: {e}")
            state["error_message"] = f"Context retrieval failed: {str(e)}"
            state["context"] = "An error occurred while retrieving context."
            return state

    async def generate_and_decide(self, state: GraphState) -> GraphState:
        """Generate a response or decide on the next action in a single step."""
        print(f"GENERATE AND DECIDE for ticket {state['ticket_id']}, {state["messages"]}")

        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are 'Masai Agent', an AI support expert for Masai School. Your task is to analyze the user's query and available context to make a single, definitive decision.

**Decision Logic:**
1.  **Classify Team:** First, determine if the query is for **EC (Experience Champion)** or **IA (Instructor Associate)**.
    * **EC:** Logistics, attendance, leave, evaluations, placements, finances (ISA/NBFC), non-technical queries.
    * **IA:** Technical issues, coding, DSA, code reviews, academic doubts.

2.  **Choose One Action:**
    a. **REQUEST INFO:** If the query lacks specific details needed for a full answer (e.g., missing dates, specifics of a bug), choose this. You MUST list the needed info in `missing_info`.
    b. **ESCALATE:** If the query is too complex, sensitive, or requires a manual action you can't perform (e.g., "the video link is broken"), choose this. You MUST provide a clear `escalation_reason`.
    c. **RESPOND:** If you have enough context to fully and accurately answer, choose this. You MUST generate a helpful and complete `response`.

**Output Format:**
Respond ONLY with 1 valid JSON object matching the `AgentDecision` schema as follows. Do not add explanations or markdown or any Markdown fences (```json ...``` or ```).
IMPORTANT: Output must be exactly one JSON object. 
Do not repeat the JSON object or KEY - VALUES in the object. 
Do not output multiple decision objects. 
If you output anything else, it will cause a system error.

AgentDecision Schema = {{
  "decision": "respond" | "request_info" | "escalate",
  "response": "Your generated response here. Null if not applicable.",
  "missing_info": ["List of questions or items needed. Null if not applicable."],
  "escalation_reason": "Reason for escalation. Null if not applicable.",
  "admin_type": "EC" | "IA",
  "confidence": "Your confidence score in the range [0.0, 1.0]. Must be a float."
}}"""),

            MessagesPlaceholder(variable_name="messages"),
            ("human", """Here is the data. Make your decision.
**Available Knowledge Base Context (Previously resolved tickets for reference / Program and Curriculum details):**
{context}

**Current User Query:**
{query}
""")
        ])
        
        chain = prompt | self.llm
        try:
            result = await chain.ainvoke({
                "messages": state["messages"],
                "context": state["context"],
                "query": state["original_query"]
            })
            
            raw = result.content.strip()
            print(f"LLM ROUTING RESPONSE (raw):\n{raw}\n")

            # 1. Strip any Markdown fences (```json ...``` or ```)
            if raw.startswith("```"):
                parts = raw.split("```")
                raw = parts[1] if len(parts) > 1 else parts[0]
                if raw.startswith("json\n"):
                    raw = raw[5:]

            # 2. Extract the first {...} block in case there’s any trailing text
            # re.DOTALL ensures '.' matches newlines
            match = re.search(r"\{.*\}", raw, re.DOTALL)
            if match:
                json_str = match.group(0)
            else:
                json_str = raw
            
            decision_json = json.loads(json_str)
            
            # Validate with Pydantic
            agent_decision = AgentDecision(**decision_json)
            state["agent_decision"] = agent_decision.model_dump()
            print(f"Decision: {state['agent_decision']['decision']}, Team: {state['agent_decision']['admin_type']}")

        except Exception as e:
            print(f"GENERATE/DECIDE ERROR: {e}\nContent was: {getattr(result, 'content', 'N/A')}")
            traceback.print_exc()
            state["error_message"] = f"LLM decision failed: {str(e)}"
            # Fallback to a safe escalation
            state["agent_decision"] = {
                "decision": "escalate", "escalation_reason": "AI agent encountered a processing error.",
                "admin_type": "EC", "confidence": 0.0, "response": None, "missing_info": None
            }
        
        state["steps_taken"].append("decision_made")
        return state

    async def finalize_and_act(self, state: GraphState) -> GraphState:
        """Executes the decision made by the agent and updates the ticket."""
        print(f"FINALIZING ticket {state['ticket_id']}")
        try:
            decision = state.get("agent_decision")
            ticket_id = state["ticket_id"]

            if not decision:
                raise ValueError("Agent decision not found in state, cannot finalize.")

            action = decision.get("decision").lower().replace(" ", "_")
            confidence = float(decision.get("confidence", 0.0))

            # Outcome 1: Request more information from the student
            if action == 'request_info' and decision.get('missing_info'):
                print("Action: Requesting info from student.", decision.get("admin_type", "EC"), {state['agent_decision']['admin_type']})
                message = "Thank you for contacting us. To better assist you, could you please provide the following information?\n\n" + "\n".join(f"• {info}" for info in decision['missing_info'])
                conversation_service.create_conversation(ticket_id, "agent", message, confidence_score=confidence)
                
                admin = await self._find_available_admin(decision.get("admin_type", "EC"))
                print(f"admin in request info {admin}.")
                admin_id = admin["id"] if admin else None
                print(f"assigning admin in request info {admin_id}.")
                ticket_service.update_ticket_status(ticket_id, TicketStatus.STUDENT_ACTION_REQUIRED.value, admin_id)
                state["final_status"] = TicketStatus.STUDENT_ACTION_REQUIRED.value

            # Outcome 2: Escalate to a human admin
            elif action == 'escalate':
                print(f"Action: Escalating to human admin ({decision.get('admin_type')}).")
                # The escalation agent assigns the ticket, sets status, and creates the conversation
                await self.escalation_agent.process({
                    "ticket_id": ticket_id, "admin_type": decision.get("admin_type", "EC")
                })
                state["final_status"] = TicketStatus.ADMIN_ACTION_REQUIRED.value

            # Outcome 3: Respond to the student and resolve the ticket
            elif action == 'respond' and decision.get('response'):
                print("Action: Responding and resolving ticket.",decision.get("admin_type", "EC"), )
                response = decision['response']
                conversation_service.create_conversation(ticket_id, "agent", response, confidence_score=confidence)
                
                admin = await self._find_available_admin(decision.get("admin_type", "EC"))
                print(f"admin in respond {admin}.")
                admin_id = admin["id"] if admin else None
                print(f"assigning admin in respond {admin_id}.")
                ticket_service.update_ticket_status(ticket_id, TicketStatus.RESOLVED.value, admin_id)
                state["final_status"] = TicketStatus.RESOLVED.value
                
                # Cache high-confidence, successful responses
                if confidence >= 0.85:
                    await self.cache_service.store_response(state["original_query"], response, confidence, state["category"])
                    print("Stored successful response in cache.")
            
            else:
                raise ValueError(f"Invalid or incomplete agent decision: {action}")

        except Exception as e:
            print(f"FINALIZATION ERROR: {e}")
            traceback.print_exc()
            state["error_message"] = f"Finalization failed: {str(e)}"
            # Safe fallback: escalate the ticket
            await self.escalation_agent.process({"ticket_id": state["ticket_id"], "admin_type": "EC"})
            state["final_status"] = TicketStatus.ADMIN_ACTION_REQUIRED.value

        state["steps_taken"].append(f"finalized_as_{state['final_status']}")
        return state

    async def process_ticket(self, ticket_id: str):
        """Main entry point to process a ticket through the workflow."""
        print(f"\n--- STARTING WORKFLOW for ticket {ticket_id} ---")
        initial_state = GraphState(ticket_id=ticket_id)
        try:
            final_state = await self.workflow.ainvoke(initial_state)
            print(f"--- WORKFLOW COMPLETED for ticket {ticket_id}: Final Status = {final_state.get('final_status')} ---")
            return final_state
        except Exception as e:
            print(f"--- WORKFLOW EXECUTION ERROR for ticket {ticket_id}: {e} ---")
            traceback.print_exc()
            raise

# Export for use in other modules
workflow_instance = EnhancedLangGraphWorkflow()

async def process_ticket_async(ticket_id: str):
    """Async wrapper for background task execution."""
    print(f"Processing ticket {ticket_id} through LangGraph workflow...")
    await workflow_instance.process_ticket(ticket_id)