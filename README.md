# Intelligent_lms_support
Automating LMS Support Ticket Resolution Using AI

## Commands

### Development Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your API keys (Google AI, Pinecone, etc.)

# Start the system
python run.py
```

### Database Operations
```bash
# Initialize database with sample users
python backend/app/db/init_db.py
```

### API Testing
- API Documentation: http://localhost:8000/docs
- Sample Student Login: student1@masaischool.com / password123  
- Sample Admin Login: ec1@masaischool.com / admin123

## Knowledge Base Categories

The system uses exactly 3 knowledge base categories mapped to MongoDB collections:

1. **Program Details** (`program_details_documents`) - Course info, schedules, structure
2. **Q&A** (`qa_documents`) - FAQs, troubleshooting, general support  
3. **Curriculum Documents** (`curriculum_documents`) - Technical content, assignments, evaluations

## Key Implementation Details

### Backend Structure
- **FastAPI**: REST API with session-based authentication (no JWT)
- **MongoDB**: All data storage including:
  - `users` collection: User accounts and authentication
  - `tickets` collection: Support tickets with metadata
  - `conversations` collection: Ticket conversation threads  
  - `program_details_documents` collection: Course info, schedules
  - `qa_documents` collection: FAQs, troubleshooting
  - `curriculum_documents` collection: Technical content, assignments
- **Pinecone**: Vector database for RAG retrieval
- **Redis**: Semantic caching system
- **LangGraph**: Multi-agent workflow orchestration

### Multi-Agent Workflow
1. **Routing Agent**: Cache check, query triage (EC/IA), decomposition
2. **Retriever Agent**: Vector search with knowledge base category filtering
3. **Response Agent**: LLM response generation with confidence scoring
4. **Escalation Agent**: Human handoff when confidence < 85%

### API Endpoints Structure
- `/v1/auth/*` - Authentication (login/logout, session management)
- `/v1/tickets/*` - Student operations (create, list, detail, rate, reopen)
- `/v1/admin/*` - Admin operations (ticket management, document upload/delete)

### Environment Variables Required
- `GOOGLE_API_KEY` - For Gemini LLM and embeddings
- `PINECONE_API_KEY` - Vector database access
- `MONGODB_URL` - MongoDB connection (single database for all collections)
- `REDIS_URL` - Redis connection
- `SESSION_SECRET_KEY` - For secure session cookies