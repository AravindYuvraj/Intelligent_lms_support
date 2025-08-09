from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File, Form
from typing import List, Optional, Dict, Any
from backend.app.models import user_service, ticket_service, conversation_service, TicketStatus
from backend.app.core.deps import get_current_admin, get_document_service
from backend.app.api.tickets.schemas import TicketListResponse, TicketDetailResponse, ConversationResponse, TicketResponse
from backend.app.services.document_service import DocumentService
import logging

logger = logging.getLogger(__name__)
router = APIRouter()

# --------------------------------------------------------------------------
# NOTE on Ticket Listing:
# The original code had an "N+1" query problem, causing many database calls.
# This improved version assumes the ticket_service has a more efficient method
# like `get_admin_tickets_with_details` that fetches all required data 
# (ticket info, conversation counts, user details) in a single, optimized query.
# --------------------------------------------------------------------------
@router.get("/tickets", response_model=List[TicketListResponse])
async def get_admin_tickets(
    status_filter: Optional[str] = None,
    admin_type: Optional[str] = None,
    current_user: Dict[str, Any] = Depends(get_current_admin)
):
    """Get all tickets that can be viewed by the admin."""
    
    # This single service call is assumed to efficiently fetch all data
    tickets_with_details = ticket_service.get_admin_tickets_with_details(
        admin_id=current_user["id"],
        admin_type=admin_type,
        status_filter=status_filter
    )
    
    # The response from the service should already be structured for the API
    return [TicketListResponse(**ticket) for ticket in tickets_with_details]

@router.post("/tickets/{ticket_id}/respond")
async def respond_to_ticket(
    ticket_id: str,
    message: str = Form(...),
    current_user: Dict[str, Any] = Depends(get_current_admin)
):
    """Admin responds to a ticket, setting status to Work in Progress"""
    
    ticket = ticket_service.get_ticket_by_id(ticket_id)
    if not ticket:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Ticket not found"
        )
    
    # Update ticket status to Work in Progress
    new_status = TicketStatus.WIP.value
    ticket_service.update_ticket_status(ticket_id, new_status, current_user["id"])
    
    conversation_service.create_conversation(
        ticket_id=ticket_id,
        sender_role="admin",
        sender_id=current_user["id"],
        message=message
    )
    
    return {
        "message": "Response submitted successfully",
        "ticket_status": new_status
    }

@router.post("/tickets/{ticket_id}/resolve")
async def resolve_ticket(
    ticket_id: str,
    message: str = Form(...),
    current_user: Dict[str, Any] = Depends(get_current_admin)
):
    """Admin resolves a ticket"""
    
    # Get ticket
    ticket = ticket_service.get_ticket_by_id(ticket_id)
    if not ticket:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Ticket not found"
        )
    
    # Update ticket status to Resolved
    new_status = TicketStatus.RESOLVED.value
    ticket_service.update_ticket_status(ticket_id, new_status, current_user["id"])
    
    # Add conversation entry
    conversation_service.create_conversation(
        ticket_id=ticket_id,
        sender_role="admin",
        sender_id=current_user["id"],
        message=message
    )
    
    try:
        # Get original query from first conversation
        conversations = conversation_service.get_ticket_conversations(ticket_id)
        original_conv = next((c for c in conversations if c["sender_role"] == "student"), None)
        
        if original_conv:
            from backend.app.agents.cache_service import SemanticCacheService
            cache_service = SemanticCacheService()
            await cache_service.store_response(
                query=original_conv["message"],
                response=message,
                confidence=0.95,  # High confidence for human responses
                category=ticket["category"]
            )
    except Exception as e:
        logger.error(f"Error storing admin response in cache: {str(e)}")
    
    return {
        "message": "Ticket resolved successfully",
        "ticket_status": new_status
    }

@router.post("/documents/upload")
async def upload_document(
    category: str = Form(...),
    file: UploadFile = File(...),
    document_service: DocumentService = Depends(get_document_service),
    current_user: Dict[str, Any] = Depends(get_current_admin)
):
    """Upload a document to the knowledge base."""
    logger.info(f"Uploading document '{file.filename}' to category '{category}'.")
    try:
        # Use the injected service instance. Validation is handled by the service.
        result = await document_service.upload_document(file, category)
        return {
            "message": "Document uploaded successfully",
            "document_id": result["document_id"],
            "category": category,
            "items_created": result["items_created"]
        }
    except ValueError as e: # Catch specific validation errors from the service
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Document upload error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to upload document: {str(e)}"
        )

@router.delete("/documents/{doc_id}")
async def delete_document(
    doc_id: str,
    document_service: DocumentService = Depends(get_document_service),
    current_user: Dict[str, Any] = Depends(get_current_admin)
):
    """Delete a document from the knowledge base."""
    try:
        # Use the injected service instance
        await document_service.delete_document(doc_id)
        return {"message": "Document deleted successfully"}
    except ValueError as e: # Catch not found error
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Document deletion error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete document: {str(e)}"
        )

@router.get("/documents")
async def list_documents(
    category: Optional[str] = None,
    document_service: DocumentService = Depends(get_document_service),
    current_user: Dict[str, Any] = Depends(get_current_admin)
):
    """List documents in the knowledge base."""
    try:
        # Use the injected service instance. Validation for category is handled inside.
        documents = await document_service.list_documents(category)
        # Get the list of valid categories directly from the service
        valid_categories = list(document_service.valid_categories)
        return {"documents": documents, "categories": valid_categories}
    except Exception as e:
        logger.error(f"Document listing error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list documents: {str(e)}"
        )