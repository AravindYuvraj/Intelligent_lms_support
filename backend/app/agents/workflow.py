from typing import Dict, Any
from backend.app.models import ticket_service, conversation_service, user_service, TicketStatus, UserRole
from .state import AgentState, WorkflowStep
from .routing_agent import RoutingAgent
from .retriever_agent import RetrieverAgent
from .response_agent import ResponseAgent
from .escalation_agent import EscalationAgent
import logging

logger = logging.getLogger(__name__)

class TicketWorkflow:
    def __init__(self):
        self.routing_agent = RoutingAgent()
        self.retriever_agent = RetrieverAgent()
        self.response_agent = ResponseAgent()
        self.escalation_agent = EscalationAgent()
    
    def run(self, ticket_id: str):
        """Main workflow orchestrator - synchronous wrapper for async operations"""
        import asyncio
        
        try:
            # Run the async workflow
            asyncio.run(self._run_async(ticket_id))
        except Exception as e:
            logger.error(f"Workflow execution error for ticket {ticket_id}: {str(e)}")
            self._handle_workflow_error(ticket_id, str(e))
    
    async def _run_async(self, ticket_id: str):
        """Async workflow execution"""
        try:
            # Get ticket and initialize state
            ticket = ticket_service.get_ticket_by_id(ticket_id)
            if not ticket:
                raise ValueError(f"Ticket {ticket_id} not found")
            
            # Get user information
            user = user_service.get_user_by_id(ticket["user_id"])
            if not user:
                raise ValueError(f"User for ticket {ticket_id} not found")
            
            # Get conversation history
            conversations = conversation_service.get_ticket_conversations(ticket_id)
            
            # Initialize state
            state = AgentState(
                ticket_id=ticket_id,
                user_id=ticket["user_id"],
                query=ticket["message"],  # Use original ticket message
                category=ticket["category"],
                subcategory_data=ticket.get("subcategory_data"),
                attachments=ticket.get("attachments"),
                current_step=WorkflowStep.ROUTING.value,
                confidence_score=None,
                response=None,
                cached_response=None,
                retrieved_context=None,
                admin_type=None,
                requires_escalation=False,
                missing_information=None,
                conversation_history=[{
                    "role": conv["sender_role"],
                    "message": conv["message"],
                    "timestamp": conv["timestamp"].isoformat()
                } for conv in conversations],
                ticket_status=ticket["status"],
                error_message=None
            )
            
            # Execute workflow steps
            max_iterations = 10
            iterations = 0
            
            while state["current_step"] != WorkflowStep.COMPLETION.value and iterations < max_iterations:
                iterations += 1
                logger.info(f"Ticket {ticket_id} - Step {iterations}: {state['current_step']}")
                
                if state["current_step"] == WorkflowStep.ROUTING.value:
                    state = await self.routing_agent.process(state)
                
                elif state["current_step"] == WorkflowStep.RETRIEVAL.value:
                    state = await self.retriever_agent.process(state)
                
                elif state["current_step"] == WorkflowStep.RESPONSE_GENERATION.value:
                    state = await self.response_agent.process(state)
                
                elif state["current_step"] == WorkflowStep.ESCALATION.value:
                    state = await self.escalation_agent.process(state)
                
                else:
                    logger.warning(f"Unknown workflow step: {state['current_step']}")
                    break
            
            # Final processing
            await self._finalize_ticket(ticket_id, state)
            
            logger.info(f"Workflow completed for ticket {ticket_id} in {iterations} steps")
            
        except Exception as e:
            logger.error(f"Async workflow error for ticket {ticket_id}: {str(e)}")
            raise e
    
    async def _finalize_ticket(self, ticket_id: str, state: AgentState):
        """Finalize ticket processing"""
        try:
            # Update ticket status based on workflow outcome
            if state.get("requires_escalation") or state.get("error_message"):
                ticket_service.update_ticket_status(ticket_id, TicketStatus.ADMIN_ACTION_REQUIRED.value)
            elif state.get("response") and state.get("confidence_score", 0) >= 0.85:
                ticket_service.update_ticket_status(ticket_id, TicketStatus.RESOLVED.value)
                
                # Add final response to conversation
                conversation_service.create_conversation(
                    ticket_id=ticket_id,
                    sender_role="agent",
                    message=state["response"],
                    confidence_score=state["confidence_score"]
                )
            
            logger.info(f"Ticket {ticket_id} finalized")
            
        except Exception as e:
            logger.error(f"Finalization error for ticket {ticket_id}: {str(e)}")
    
    def _handle_workflow_error(self, ticket_id: str, error_message: str):
        """Handle workflow errors"""
        try:
            ticket_service.update_ticket_status(ticket_id, TicketStatus.ADMIN_ACTION_REQUIRED.value)
            
            # Add error message to conversation
            conversation_service.create_conversation(
                ticket_id=ticket_id,
                sender_role="agent",
                message=f"System Error: {error_message}. Your ticket has been escalated to our support team."
            )
            
        except Exception as e:
            logger.error(f"Error handling workflow error: {str(e)}")