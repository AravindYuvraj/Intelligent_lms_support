from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File, Form
from typing import List, Optional, Dict, Any
from backend.app.models import user_service, ticket_service, conversation_service, TicketStatus
from backend.app.core.deps import get_current_admin
from backend.app.api.tickets.schemas import TicketListResponse, TicketDetailResponse, ConversationResponse, TicketResponse
from backend.app.services.document_service import DocumentService
import logging

logger = logging.getLogger(__name__)
router = APIRouter()

@router.get("/tickets", response_model=List[TicketListResponse])
async def get_admin_tickets(
    status_filter: Optional[str] = None,
    current_user: Dict[str, Any] = Depends(get_current_admin)
):
    """Get all tickets that can be viewed by the admin"""
    
    # Get tickets for admin (assigned or unassigned)
    tickets = ticket_service.get_admin_tickets(current_user["id"])
    
    # Apply status filter if provided
    if status_filter:
        tickets = [t for t in tickets if t["status"] == status_filter]
    
    result = []
    for ticket in tickets:
        # Get conversation count and last response
        response_count = conversation_service.get_conversation_count(ticket["id"])
        last_conversation = conversation_service.get_last_conversation(ticket["id"])
        
        # Get student info
        student = user_service.get_user_by_id(ticket["user_id"])
        
        result.append(TicketListResponse(
            id=ticket["id"],
            user_id=ticket["user_id"],
            category=ticket["category"],
            status=ticket["status"],
            title=ticket["title"],
            created_at=ticket["created_at"],
            updated_at=ticket.get("updated_at"),
            rating=ticket.get("rating"),
            assigned_to=ticket.get("assigned_to"),
            assigned_admin_email=current_user["email"] if ticket.get("assigned_to") == current_user["id"] else None,
            response_count=response_count,
            last_response=last_conversation["message"] if last_conversation else None,
            last_response_time=last_conversation["timestamp"] if last_conversation else None
        ))
    
    return result

@router.post("/tickets/{ticket_id}/respond")
async def respond_to_ticket(
    ticket_id: str,
    message: str = Form(...),
    status: str = Form(...),  # "Resolved" or "Work in Progress"
    current_user: Dict[str, Any] = Depends(get_current_admin)
):
    """Admin responds to a ticket"""
    
    # Get ticket
    ticket = ticket_service.get_ticket_by_id(ticket_id)
    if not ticket:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Ticket not found"
        )
    
    # Validate status
    if status not in ["Resolved", "Work in Progress"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid status. Must be 'Resolved' or 'Work in Progress'"
        )
    
    # Update ticket
    new_status = TicketStatus.RESOLVED.value if status == "Resolved" else TicketStatus.WIP.value
    ticket_service.update_ticket_status(ticket_id, new_status, current_user["id"])
    
    # Add conversation entry
    conversation_service.create_conversation(
        ticket_id=ticket_id,
        sender_role="admin",
        sender_id=current_user["id"],
        message=message
    )
    
    # If resolved, add to cache and knowledge base
    if status == "Resolved":
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
        "message": "Response submitted successfully",
        "ticket_status": new_status
    }

@router.post("/documents/upload")
async def upload_document(
    category: str = Form(...),
    file: UploadFile = File(...),
    current_user: Dict[str, Any] = Depends(get_current_admin)
):
    """Upload a document to the knowledge base"""
    print("Uploading document:", file.filename, "to category:", category)
    # Validate category
    valid_categories = ["program_details_documents", "qa_documents", "curriculum_documents"]
    if category not in valid_categories:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid category. Must be one of: {valid_categories}"
        )
    
    try:
        document_service = DocumentService()
        result = await document_service.upload_document(file, category)
        print("Document upload result:", result)
        return {
            "message": "Document uploaded successfully",
            "document_id": result["document_id"],
            "category": category,
            "items_created": result["items_created"]
        }
    except Exception as e:
        logger.error(f"Document upload error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to upload document: {str(e)}"
        )

@router.delete("/documents/{doc_id}")
async def delete_document(
    doc_id: str,
    current_user: Dict[str, Any] = Depends(get_current_admin)
):
    """Delete a document from the knowledge base"""
    
    try:
        document_service = DocumentService()
        await document_service.delete_document(doc_id)
        return {"message": "Document deleted successfully"}
    except Exception as e:
        logger.error(f"Document deletion error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete document: {str(e)}"
        )

@router.get("/documents")
async def list_documents(
    category: Optional[str] = None,
    current_user: Dict[str, Any] = Depends(get_current_admin)
):
    """List documents in the knowledge base"""
    
    if category:
        valid_categories = ["program_details_documents", "qa_documents", "curriculum_documents"]
        if category not in valid_categories:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid category. Must be one of: {valid_categories}"
            )
    
    try:
        document_service = DocumentService()
        documents = await document_service.list_documents(category)
        return {"documents": documents, "categories": ["program_details_documents", "qa_documents", "curriculum_documents"]}
    except Exception as e:
        logger.error(f"Document listing error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list documents: {str(e)}"
        )