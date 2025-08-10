from typing import Optional, Dict, Any
from backend.app.models import ticket_service, conversation_service, user_service, TicketStatus

def get_kb_category(ticket_category: Optional[str]) -> Optional[str]:
        """
        Maps an incoming ticket category to one of the three main knowledge base categories.
        
        Returns the name of the knowledge base category (e.g., "program_details_documents").
        """
        if not ticket_category:
            print("No category provided, defaulting to qa_documents")
            return "qa_documents"

        # Enhanced mapping with more detailed logging
        category_mapping = {
            # Program and administrative related -> Program Details
            "Course Query": "program_details_documents",
            "Attendance/Counselling Support": "program_details_documents", 
            "Leave": "program_details_documents",
            "Late Evaluation Submission": "program_details_documents",
            "Missed Evaluation Submission": "program_details_documents",
            "Withdrawal": "program_details_documents",
            
            # Technical and curriculum related -> Curriculum Documents
            "Evaluation Score": "curriculum_documents",
            "MAC": "curriculum_documents",
            "Revision": "curriculum_documents",
            
            # General support, FAQs, troubleshooting -> qa_documents
            "Product Support": "qa_documents",
            "NBFC/ISA": "qa_documents",
            "Feedback": "qa_documents",
            "Referral": "qa_documents",
            "Personal Query": "qa_documents",
            "Code Review": "qa_documents",
            "Placement Support - Placements": "qa_documents",
            "Offer Stage- Placements": "qa_documents", 
            "ISA/EMI/NBFC/Glide Related - Placements": "qa_documents",
            "Session Support - Placement": "qa_documents",
            "IA Support": "qa_documents",
        }
        
        mapped_category = category_mapping.get(ticket_category)
        
        if mapped_category:
            print(f"CATEGORY MAPPING: '{ticket_category}' -> '{mapped_category}'")
        else:
            print(f" UNMAPPED CATEGORY: '{ticket_category}', defaulting to 'qa_documents'")
            mapped_category = "qa_documents"
        
        return mapped_category

async def find_available_admin( admin_type: str) -> Dict[str, Any]:
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
            print(f"Error finding admin: {e}")
            return None
      
