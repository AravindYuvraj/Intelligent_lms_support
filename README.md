# Intelligent LMS Support System



# 🚀 Intelligent LMS Support System

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python Version](https://img.shields.io/badge/Python-3.11%2B-blue?logo=python)](https://www.python.org/)
[![Node.js Version](https://img.shields.io/badge/Node.js-18%2B-green?logo=node.js)](https://nodejs.org/)
[![FastAPI](https://img.shields.io/badge/Backend-FastAPI-009688?logo=fastapi)](https://fastapi.tiangolo.com/)
[![Next.js](https://img.shields.io/badge/Frontend-Next.js-black?logo=next.js)](https://nextjs.org/)
[![MongoDB](https://img.shields.io/badge/Database-MongoDB-47A248?logo=mongodb)](https://www.mongodb.com/)
[![Pinecone](https://img.shields.io/badge/Vector%20DB-Pinecone-64B5F6?logo=pinecone)](https://www.pinecone.io/)
[![LangChain](https://img.shields.io/badge/AI%20Framework-LangChain-007BFF?logo=langchain)](https://www.langchain.com/)
[![LangGraph](https://img.shields.io/badge/AI%20Orchestration-LangGraph-FF5722?logo=langgraph)](https://langchain-ai.github.io/langgraph/)

---


## 📚 Table of Contents
1. [Introduction](#-introduction)  
2. [Features](#-features)  
3. [Architecture](#-architecture)  
4. [Technology Stack](#-technology-stack)  
5. [Getting Started](#-getting-started)  
6. [Usage](#-usage)  
7. [API Endpoints](#-api-endpoints)  
8. [Project Structure](#-project-structure)  
9. [Future Enhancements](#-future-enhancements)  
10. [Contributing](#-contributing)  
11. [License](#-license)  
12. [Contact](#-contact)  

---

## 📝 Introduction

The **Intelligent LMS Support System** is an **AI-powered ticketing solution** for Learning Management Systems that:  
- Automates student query resolution.  
- Retrieves knowledge from a centralized database.  
- Escalates complex cases to human admins.  

This is a production-ready AI-powered support system for Masai School's LMS that automates 70-80% of support ticket resolution using:

- **LangGraph**: Orchestrates multi-agent workflow with state management
- **LangChain**: Provides LLM integration and prompt engineering
- **Google Gemini**: Powers natural language understanding and generation
- **Pinecone**: Vector database for semantic search and RAG
- **MongoDB**: Primary data storage for tickets, users, and documents
- **Redis**: High-performance semantic caching
- **FastAPI**: Modern, fast web framework for building APIs

---

## ✨ Features


- 🤖 **AI-Powered Ticket Resolution** – Automated, accurate responses.  
- 📚 **Contextual Understanding** – Retrieval-Augmented Generation (RAG).  
- ⚡ **Semantic Caching** – Faster repeated queries.  
- 🆘 **Intelligent Escalation** – Human handoff when needed.  
- 👤 **Role-Based Access Control** – Student/Admin separation.  
- 🗂 **Knowledge Base Management**  
  - Document ingestion (PDF, Excel, CSV, etc.)  
  - Categorization  
  - Listing & deletion tools  
- 📊 **Analytics Dashboard** – Automation rates, agent performance.  
- 🔄 **Modular & Scalable** – Easy to add new programs and knowledge bases.
- 🎨 **User-Friendly UI** – Intuitive design for all roles.  

---

## 🏗 Architecture

### High-Level Overview
- **Frontend (Next.js)** – UI for students & admins.  
- **Backend (FastAPI)** – API, authentication, ticket handling.  
- **AI Core (LangChain/LangGraph)** – Query processing, retrieval, decision-making.  
- **Databases:** MongoDB, Pinecone, Redis.  

```mermaid
graph TD
    A[Student Query] --> B{Routing Agent}
    B --> C{Cache Check (Redis)}
    C -- Cache Hit --> D[Personalize Response (LLM)]
    D --> E[Response to Student]
    C -- Cache Miss --> F{Query Decomposition}
    F --> G{Triage (EC/IA)}
    G --> H{Retriever Agent}
    H --> I{Pinecone Vector DB}
    I --> J{Re-ranking}
    J --> K{Response Agent}
    K --> L{Confidence Scoring}
    L -- High Confidence (>=85%) --> E
    L -- Medium/Low Confidence (<85%) --> M{Escalation Agent}
    M --> N[Notify Admin]
    N --> O[Admin Response]
    O --> P[Update Cache & KB]
    P --> E
````

---

## 🛠 Technology Stack

🖥 Backend

* **Framework:** FastAPI (Python 3.11+)
* **DB:** MongoDB (pymongo)
* **Vector DB:** Pinecone
* **Cache/Analytics:** Redis
* **AI Frameworks:** LangChain, LangGraph
* **LLM:** Google Generative AI (Gemini 2.0 Flash)
* **Embeddings:** HuggingFace `all-mpnet-base-v2`
* **Document Processing:** unstructured.io, pandas, pdfminer.six, pytesseract
* **Auth:** passlib, python-jose
* **Deployment:** Docker, Railway


💻 Frontend

* **Framework:** Next.js 14 (React, TypeScript)
* **Styling:** Tailwind CSS + Shadcn/ui
* **HTTP Client:** axios
* **Auth:** js-cookie, jwt-decode
* **Deployment:** Railway


---

## 🚀 Getting Started

### Prerequisites

* Python 3.11+
* Node.js 18+
* MongoDB & Redis instances
* Google API Key
* Pinecone API Key

### Environment Variables

<details>
<summary>📄 Click to view .env example</summary>

```env
# Database
MONGODB_URL="mongodb://localhost:27017/lms_support"
REDIS_URL="redis://localhost:6379"

# Google AI
GOOGLE_API_KEY="YOUR_GOOGLE_API_KEY"

# Pinecone
PINECONE_API_KEY="YOUR_PINECONE_API_KEY"
PINECONE_ENVIRONMENT="YOUR_PINECONE_ENVIRONMENT"

# Index & Collection Maps
PINECONE_INDEX_MAP='{"qa_documents": "qa-documents-index"}'
MONGO_COLLECTION_MAP='{"qa_documents": "qa_documents_meta"}'

# App Config
ENVIRONMENT="development"
DEBUG=True
SESSION_SECRET_KEY="YOUR_SECRET"
```

</details>

---

## ▶ Installation

#### Backend

```bash
cd Intelligent_lms_support
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install uv
uv sync
python backend/app/db/init_db.py
```

#### Frontend

```bash
cd frontend
npm install  # or pnpm install
echo 'NEXT_PUBLIC_API_BASE="http://localhost:8000"' > .env.local
```

---

## 🏃 Running the Application

```bash
# Backend
uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --reload

# Frontend
npm run dev
```

* Backend: [http://localhost:8000](http://localhost:8000)
* Frontend: [http://localhost:3000](http://localhost:3000)

---

## 🎯 Usage

<details>
<summary>👨‍🎓 Student Flow</summary>

1. Visit `http://localhost:3000`
2. Login as student
3. Create, track, and manage tickets
4. Reopen or rate resolved tickets

</details>

<details>
<summary>🛠 Admin Flow</summary>

1. Login as admin
2. Manage tickets and documents
3. Respond or escalate tickets
4. View analytics dashboard

</details>

---

## 📡 API Endpoints


### Authentication
- `POST /v1/auth/login` - User login
- `POST /v1/auth/logout` - User logout

### Tickets (Students)
- `POST /v1/tickets/create` - Create new ticket
- `GET /v1/tickets/my_tickets` - List user's tickets
- `GET /v1/tickets/{ticket_id}` - Get ticket details
- `POST /v1/tickets/{ticket_id}/reopen` - Reopen resolved ticket
- `POST /v1/tickets/{ticket_id}/rate` - Rate ticket resolution

### Admin Operations
- `GET /v1/admin/tickets` - List tickets for admin
- `POST /v1/admin/tickets/{ticket_id}/respond` - Admin response
- `POST /v1/admin/documents/upload` - Upload knowledge base document
- `DELETE /v1/admin/documents/{doc_id}` - Delete document
- `GET /v1/admin/documents` - List documents


---

## 🧪 Testing

### Test Credentials
- **Student**: student1@masaischool.com / password123
- **Admin**: ec1@masaischool.com / admin123

### API Documentation
Once the server is running, visit:
- Interactive API Docs: http://localhost:8000/docs
- Alternative Docs: http://localhost:8000/redoc

### Sample Test Flow
```python
# 1. Login as student
POST /v1/auth/login
{
  "username": "student1@masaischool.com",
  "password": "password123"
}

# 2. Create a ticket
POST /v1/tickets/create
{
  "category": "Course Query",
  "title": "Cannot access Unit 3",
  "message": "I completed Unit 2 but cannot see Unit 3 materials"
}

# 3. Check ticket status
GET /v1/tickets/my_tickets

# The system will automatically process the ticket through the LangGraph workflow
```

## 📈 Performance Metrics

- **Response Time**: <2 seconds for cached queries, <5 seconds for new queries
- **Automation Rate**: 70-80% of tickets resolved without human intervention
- **Confidence Threshold**: 85% for automatic resolution
- **Cache Hit Rate**: ~40% after initial training period
- **Escalation Rate**: 20-30% for complex or unclear queries

## 🔧 Configuration

### Environment Variables
```env
# Required
GOOGLE_API_KEY=your_gemini_api_key
MONGODB_URL=mongodb://localhost:27017/lms_support

# Recommended
PINECONE_API_KEY=your_pinecone_key
PINECONE_INDEX_NAME=lms-support-index
REDIS_URL=redis://localhost:6379/0
```

### Adding New Programs
1. Upload program-specific FAQs via admin API
2. Upload curriculum documents
3. Update program metadata in database
4. System automatically adapts to new content

## 🐛 Troubleshooting

### Common Issues

1. **MongoDB Connection Error**
   - Ensure MongoDB is installed and running
   - Check connection string in .env

2. **API Key Errors**
   - Verify Google Gemini API key is valid
   - Check Pinecone API key and environment

3. **Import Errors**
   - Run `pip install -r requirements.txt` again
   - Check Python version (3.8+ required)

4. **Cache Not Working**
   - Install and start Redis
   - Check Redis connection URL
---
## 📂 Project Structure


```
Intelligent_lms_support/
├── backend/
│   └── app/
│       ├── agents/          # Multi-agent implementations
│       │   ├── langgraph_workflow.py  # Enhanced LangGraph workflow
│       │   ├── routing_agent.py
│       │   ├── retriever_agent.py
│       │   |
│       │   └── escalation_agent.py
│       ├── api/             # FastAPI routes
│       ├── core/            # Configuration and security
│       ├── db/              # Database initialization
│       ├── models/          # Data models
│       ├── scripts/         # Data ingestion scripts
│       └── services/        # Document and cache services
├── Documents/               # Sample FAQs and documents
├── .env                     # Environment configuration
├── requirements.txt         # Python dependencies
├── run.py                  # Main application entry
```

---

## 🔮 Future Enhancements

* LLM fine-tuning with feedback loop
* Granular analytics
* Real-time notifications
* Multi-language support

---

## 📜 License

This project is developed for Masai School as part of the LMS Support System initiative.

---

## 📬 Contact

** Aravind Yuraj / Team **
📧 [your.email@example.com](mailto:your.email@example.com)
🔗 [GitHub Profile](https://github.com/your-github-profile)

```

---

**Built with ❤️ using LangChain, LangGraph, and modern AI technologies**
