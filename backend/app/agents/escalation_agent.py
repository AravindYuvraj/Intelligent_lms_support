from typing import Dict, Any
from backend.app.models import ticket_service, conversation_service, user_service, TicketStatus, UserRole
from .state import AgentState, WorkflowStep
import logging

logger = logging.getLogger(__name__)

class EscalationAgent:
    def __init__(self):
        pass
    
    async def process(self, state: AgentState) -> AgentState:
        """Handle escalation to human admins"""
        try:
            ticket_id = state["ticket_id"]
            
            # Get ticket
            ticket = ticket_service.get_ticket_by_id(ticket_id)
            if not ticket:
                raise ValueError(f"Ticket {ticket_id} not found")
            
            # Find available admin based on admin_type (EC or IA)
            admin = await self._find_available_admin(state.get("admin_type", "EC"))
            
            if admin:
                # Assign ticket to admin
                ticket_service.update_ticket_status(
                    ticket_id, 
                    TicketStatus.ADMIN_ACTION_REQUIRED.value, 
                    admin["id"]
                )

                # TODO: Send notification to admin (implement notification system)
                await self._notify_admin(admin, ticket, state)
                
                logger.info(f"Ticket {ticket_id} escalated to admin {admin['email']}")
            else:
                # No admin available - mark as action required for any admin
                ticket_service.update_ticket_status(ticket_id, TicketStatus.ADMIN_ACTION_REQUIRED.value)
                logger.warning(f"No available admin found for ticket {ticket_id}")
            
            # Send generic message to student
            if state.get("missing_information"):
                ticket_service.update_ticket_status(
                    ticket_id, 
                    TicketStatus.STUDENT_ACTION_REQUIRED.value
                )
            student_message = await self._create_student_message(state)
            conversation_service.create_conversation(
                ticket_id=ticket_id,
                sender_role="agent",
                message=student_message
            )
            
            state["current_step"] = WorkflowStep.COMPLETION.value
            return state
                
        except Exception as e:
            logger.error(f"Error in escalation agent: {str(e)}")
            state["error_message"] = str(e)
            return state
    
    async def _find_available_admin(self, admin_type: str) -> Dict[str, Any]:
        """Find available admin of the specified type"""
        try:
            # Get all admin users
            admins = user_service.get_admins()
            
            # For now, just return the first admin
            # In production, you might implement load balancing, availability checks, etc.
            return admins[0] if admins else None
            
        except Exception as e:
            logger.error(f"Error finding admin: {str(e)}")
            return None
    
    async def _create_student_message(self, state: AgentState) -> str:
        """Create generic message for student"""
        try:
            if state.get("missing_information"):
                return (
                    "Thank you for contacting support. To help us assist you better, please provide the following information:\n\n"
                    + "\n".join(f"• {info}" for info in state["missing_information"]) +
                    "\n\nOnce you provide this information, we'll be able to help you more effectively."
                )
            else:
                return (
                    "Thank you for contacting support. Your query has been forwarded to our specialized team. "
                    "One of our experts will review your request and get back to you soon with a detailed response."
                )
                
        except Exception as e:
            logger.error(f"Error creating student message: {str(e)}")
            return "Thank you for contacting support. We're looking into your query and will get back to you soon."
    
    async def _notify_admin(self, admin: Dict[str, Any], ticket: Dict[str, Any], state: AgentState):
        """Send notification to admin (placeholder for notification system)"""
        try:
            # TODO: Implement real-time notification system
            # This could be WebSocket, email, Slack, etc.
            logger.info(f"NOTIFICATION: Admin {admin['email']} assigned to ticket {ticket['id']}")
            
        except Exception as e:
            logger.error(f"Notification error: {str(e)}")