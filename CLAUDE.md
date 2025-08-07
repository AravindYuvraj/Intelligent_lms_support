# PRD
Product Requirements Document: Masai LMS Support System
1. Introduction
The goal of this project is to develop an intelligent support system for the Masai LMS platform. The system will leverage a multi-agentic RAG (Retrieval-Augmented Generation) architecture to efficiently handle student queries and automate ticket resolution. The system will feature distinct user roles for students and admins (EC/IA), each with a dedicated set of functionalities.
2. Core Functionality and Features
2.1. Student User Flow
Login: Students can log in using their pre-assigned email and password. The system will maintain a persistent session after login.
Dashboard: A placeholder dashboard will be the entry point after login.
Support Section: Students can raise new tickets, view a list of their existing tickets, and track their status (Open, Work in Progress, Action Required, Resolved).
Ticket Detail: Clicking on a ticket will open a detail page showing the full conversation history. Students can reopen a resolved ticket from this page.
Rating: Students can rate the resolution of a ticket, which will update the database and the knowledge base.
2.2. Admin User Flow
Login: Admins (EC/IA) can log in with their assigned email and password. The session will persist upon reload.
Dashboard: A dashboard will provide an overview of their responsibilities.
Support Section: Admins will see a list of all tickets relevant to their role (EC or IA).
Ticket Detail: Admins can view the complete conversation history of a ticket. They can respond to tickets, mark them as "Resolved" or "Work in Progress," and see all queries, including those answered by an agent.
Document Management: Admins can upload, replace, or delete documents in the chosen knowledge base category. This will automatically trigger an update to the Pinecone vector database to reflect the changes with correct metadata.
Notifications: The system will notify a logged-in admin of a newly assigned ticket via a notification in the app. A notification icon in the header will show a badge count for new notifications, which can be cleared upon viewing.
Caching: Admins will have their session cached using Redis, but the system will use a semantic search for similar queries rather than a simple keyword search. This allows the LLM to pull and personalize cached responses.

3. Agentic RAG System Design
The core of the system is a multi-agentic RAG workflow orchestrated by LangGraph, as defined in the provided architecture.
Query Ingestion: A new ticket query is submitted by the student.
Routing Agent: The query is first sent to the Routing Agent.
Cache Check: The agent performs a semantic search on the Redis cache. If a similar query exists, the cached response is retrieved and sent to the LLM for personalization.
Query Decompose: If no suitable cache entry is found, the query is broken down.
Triage: The agent determines if the query is for an EC or IA admin and marks the ticket accordingly in the database.
Missing Information: If the query is unclear or requires more information, the agent outputs a list of required details, marks the ticket as "Action Required" for the student, and "WIP" (Work in Progress) for admins.
Forwarding: Otherwise, the query is forwarded to the Retriever Agent with the correct knowledge base category reference.
Retriever Agent: This agent searches the knowledge base to find relevant information.
Retrieval: It retrieves the top k chunks of information related to the query from the Pinecone vector database.
Re-ranking: The retrieved chunks are re-ranked based on their context and the rating from previously resolved tickets.
Response Agent: This agent generates the response.
Generation: It generates a response using the user's query and the retrieved context.
Confidence Scoring: It generates a confidence score for its response.
Response Handling:
High Confidence (≥85%): The response is outputted to the student, the ticket is marked "Resolved," and the response is cached and added to the knowledge base.
Medium Confidence (≥50%<85%): The ticket is sent to the Escalation Agent along with a suggested response, and a generic message is sent to the student.
Low Confidence (<50%): The ticket is immediately sent to the Escalation Agent, and a generic message is sent to the student.
Escalation Agent: This agent handles all human handoffs.
Notification: It notifies the relevant EC or IA admin via the internal app notification system. If there is a suggested response it will show the same as well with an option to edit or confirm the response and send it.
Status Update: It marks the ticket as "Action Required" for the admin.
Continuous Learning: When an admin resolves an escalated ticket, the new resolution is added to the cache and the knowledge base to improve future automation.
Multimodal LLM: The system will be able to handle multimodal queries (e.g., images) and understand Hinglish or unclear queries.
Queueing: Tickets will be processed using a queue to manage the load.

4. Technology Stack
Backend: Python with FastAPI.
Frontend: React.
LLM: Gemini 2.0 Flash for text generation. The system should be capable of using a multimodal LLM.
Embedding Model: Gemini latest embedding model.
Vector Database: Pinecone.
Database:
Users & Tickets: PostgreSQL.
Document Storage: MongoDB, with a separate collection for each knowledge base category.
Caching: Redis for semantic caching.
Frameworks: LangChain, LangGraph, and LangSmith.
File Storage: Cloudinary for ticket attachments.

5. Edge Cases and Considerations
Session Persistence: Student and Admin sessions will persist across reloads without requiring re-authentication.
Authentication: The login system will be email/password-based, with no password reset or email verification required, as all users will be pre-registered.
Ticket Reopening: Students have the ability to reopen a resolved ticket, which will be logged as a new interaction for the multi-agent system to process.
Document Management: When an admin replaces or deletes a document, the system must immediately reflect these changes in the Pinecone vector database.
Unclear Queries: The system will use its understanding of Hinglish and broken queries to attempt a resolution. If the query is too ambiguous, the system will ask for more information from the student and set the ticket status to "Action Required" for the student.

6. Implementation Plan
This project's tight deadline requires a highly focused and agile approach. The implementation will need to be streamlined, prioritizing the core agentic RAG loop and essential user flows. A minimal viable product (MVP) should be the immediate goal, followed by rapid iteration.
Phase 1: Setup and Core Backend
Tech Stack Setup: Set up the project environment with Python, FastAPI, React, PostgreSQL, MongoDB, Pinecone, Redis, and Cloudinary.
Core Logic: Implement the basic LangGraph agent flow (Routing, Retriever, Response).
Database Schema: Define and implement the database schemas for users, tickets, and knowledge bases.
Phase 2: Frontend and Integrations
User Interfaces: Develop the React components for student and admin login, ticket submission, and the ticket list view.
API Endpoints: Create the necessary FastAPI endpoints to connect the frontend to the agentic backend.
Integrations: Implement the Redis semantic caching and Pinecone vector store integrations.
Phase 3: Escalation and Deployment
Escalation Logic: Implement the Escalation Agent logic, including the in-app notification system.
Document Management: Develop the admin UI and backend logic for uploading and deleting documents, ensuring the Pinecone database is updated.
Deployment: Prepare the application for deployment to a production environment.
Testing: Perform end-to-end testing of the core student and admin flows to ensure functionality before deployment.

# Technical Architecture
Technical Architecture
This detailed technical architecture outlines the role of each library and framework in the Masai LMS Support System. The system is built on a multi-agent RAG (Retrieval-Augmented Generation) architecture, orchestrated with LangGraph.

1. Backend: Python with FastAPI
FastAPI will serve as the API gateway and the core backend framework. It will handle:
Authentication: Simple email/password login for students and admins.
API Endpoints: Serving the frontend with endpoints for ticket submission, retrieving ticket lists, fetching ticket details, and managing documents and attachments.
Request Handling: Processing incoming student queries and routing them to the LangGraph workflow.
Notification Webhooks: Triggering in-app notifications for admins when a ticket is assigned or updated.

2. Frontend: React
React will be used to build the user-facing interfaces for both students and admins.
Student Portal: Includes a dashboard, a support section to raise new tickets, and a page to view and rate ticket resolutions.
Admin Dashboard: Features a dashboard for admins, a support section to manage tickets, and a document management interface.
Real-time Notifications: The frontend will handle displaying notifications in the header with a badge count, routing admins to tickets when they click on a notification.

3. Agentic RAG System: LangChain, LangGraph & Gemini
This is the core of the system, where LangGraph orchestrates the flow of various LangChain agents. Gemini 2.0 Flash will serve as the primary LLM, with Gemini's latest embedding model for embeddings.
LangGraph: Acts as the state manager and workflow orchestrator. It defines the sequence of agents and the conditional logic for transitioning between them based on the ticket's state. The workflow will be a state graph that manages the ticket's journey from query to resolution.
LangChain: Provides the individual building blocks (agents) that LangGraph connects.
Routing Agent: This is the first agent in the chain. It uses the LLM to classify the query (e.g., EC or IA related) and determines the next step. It also handles initial cache checks and query decomposition.
Retriever Agent: This agent interacts with Pinecone. It uses the LLM to retrieve top 'k' chunks from the vector database and re-ranks them based on context and rating.
Response Agent: This agent takes the user's query and the retrieved context to generate a response using the Gemini 2.0 Flash LLM. It also generates a confidence score to decide if human intervention is needed.
Escalation Agent: This agent is triggered by the Response Agent if the confidence score is too low or if the query requires human intervention. It notifies the relevant admin and marks the ticket for action.
Gemini LLM:
Gemini 2.0 Flash: The primary LLM for generating responses, decomposing queries, and classifying intent. Its multimodal and Hinglish understanding capabilities will be used to handle complex queries.
Gemini latest embedding model: Used to convert text from documents and user queries into vector embeddings for storage in Pinecone and for semantic search in Redis.

4. Data Storage and Caching
PostgreSQL: The primary transactional database. It will store user data (students and admins) and ticket metadata (e.g., ticket title, status, conversation history).
MongoDB: Used for document storage. Each knowledge base category will have its own collection. Admins' document uploads and deletions will update the appropriate MongoDB collections.
Pinecone: The vector database for the agentic RAG system. It will store vector embeddings generated by the Gemini embedding model. Document changes in MongoDB will trigger immediate re-embedding and updating of the Pinecone index to ensure the knowledge base is always current.
Redis: Used for a semantic cache. Before a query goes through the full RAG pipeline, the Routing Agent will perform a semantic search on Redis. If a similar query and response exist, the response is retrieved and personalized by the LLM, bypassing the retrieval process to improve efficiency.
Cloudinary: A cloud-based service for storing ticket attachments uploaded by students.

5. Other Tools
LangSmith: This tool will be used for monitoring, tracing, and debugging the LangGraph workflow, allowing for better observability and continuous improvement of the agents' performance.
Celery (Implicit): Given the tight 3-day deadline and the need for a scalable queue to process tickets, a tool like Celery would be essential for handling asynchronous tasks, such as document processing and vector database updates, to avoid blocking the main API thread.


# Backend Plan
Backend Plan: API Endpoints, Schema, and Agentic Flow
This detailed backend plan outlines the architecture, data flow, and specific technical components for the Masai LMS Support System. It covers API endpoints, database schemas, the multi-agentic RAG workflow, and how information is structured and processed at each stage.

1. API Endpoints
The backend will be built with FastAPI and will expose the following endpoints:
1.1. Authentication & Users
POST /auth/login: Authenticates a user (student or admin) with email and password.
Request Body: {"email": "user@example.com", "password": "password"}
Response: {"message": "Login successful", "role": "student/admin"}
GET /users/me: Fetches the logged-in user's details.
1.2. Student Endpoints
POST /tickets/create: Creates a new ticket.
Request Body: {"category": "Product Support", "title": "Access Issue", "message": "Can't log in to OJ", "attachments": ["url1", "url2"]}
Response: {"message": "Ticket submitted successfully", "ticket_id": "TKT-123"}
GET /tickets/my_tickets: Retrieves a list of all tickets submitted by the logged-in student.
GET /tickets/{ticket_id}: Retrieves the full conversation history for a specific ticket.
POST /tickets/{ticket_id}/reopen: Reopens a resolved ticket.
POST /tickets/{ticket_id}/rate: Submits a rating for a resolved ticket.
1.3. Admin Endpoints
GET /admin/tickets: Retrieves a list of all tickets assigned to or viewable by the admin.
POST /admin/tickets/{ticket_id}/respond: Allows an admin to respond to a ticket.
Request Body: {"message": "Hello, I have looked into your query...", "status": "Resolved"}
POST /admin/documents/upload: Uploads a new document to a specific knowledge base category.
DELETE /admin/documents/{doc_id}: Deletes a document, triggering an update to the Pinecone vector store.

2. Database Schemas
PostgreSQL:
users table: id, email, password_hash, role (student/admin), created_at.
tickets table: id, user_id, category, status, title, message, created_at, assigned_to (admin_id), rating.
conversations table: id, ticket_id, sender_role (student/agent/admin), message, timestamp.
MongoDB: Each knowledge base category (e.g., product_support, leave) will have its own collection.
product_support collection: doc_id, file_name, content, metadata (e.g., category, source_url), created_at.

3. Agentic RAG Flow and Information Pipeline
The entire backend logic revolves around a LangGraph workflow that manages the state of each ticket.
Initial Input: A student's query arrives via the /tickets/create endpoint. It is packaged into a state object for the LangGraph workflow.
Template: The initial state object will look like:
JSON
{
  "query": "I can't log in to OJ. My username is 'student123'.",
  "ticket_id": "TKT-123",
  "user_id": "USR-456",
  "attachments": ["url1", "url2"],
  "status": "Open",
  "conversation_history": []
}


Routing Agent:
LLM Model: Gemini 2.0 Flash.
Process: The agent first checks the Redis semantic cache. It embeds the new query using the Gemini embedding model and searches for similar query vectors in Redis.
Cache Hit: If a similar query exists, the cached response and its metadata are retrieved. The agent then passes the original query and the cached response to the LLM for personalization. The ticket is resolved and the response is sent to the user, bypassing the full RAG pipeline.
Cache Miss: The agent uses the LLM to perform two key tasks:
Triage: Determine if the query is for an EC or IA admin.
Decomposition: Break down the query into its core components (e.g., "login issue", "OJ platform"). This decomposed query is then passed to the Retriever Agent.
Retriever Agent:
Library: The retrieval process will be handled by LlamaIndex, which provides advanced data structuring and retrieval capabilities.
Process: The agent uses the decomposed query to search the Pinecone vector database. It retrieves the top k most relevant document chunks.
Reranking: The retrieved chunks are then re-ranked based on their relevance and a confidence/rating score derived from past ticket resolutions. This ensures the most helpful information is prioritized.
Response Agent:
LLM Model: Gemini 2.0 Flash.
Process: The agent takes the original query and the re-ranked document chunks. The LLM generates a cohesive response. Simultaneously, it generates a confidence score based on how well the retrieved context answers the query.
Output Flow:
High Confidence (geq85): The response is stored in the database and sent to the student. The ticket status is updated to "Resolved," and the conversation is added to the Redis cache and Pinecone knowledge base for future use.
Medium/Low Confidence ($\< 85%$): The original query and the generated response are passed to the Escalation Agent.
Escalation Agent:
Process: This agent is responsible for human handoff. It notifies the relevant EC or IA admin via a webhook that triggers a frontend notification.
Notification: The notification will contain the ticket ID and a brief summary.
Database Update: The ticket status is set to "Action Required," and the admin's ID is stored in the assigned_to field in the tickets table.
Final Response:
Automated: If the ticket is resolved by an agent, the final response message is pushed to the student's conversation history via a webhook or push notification.
Admin-Handled: If an admin resolves the ticket, their response is stored in the conversation history and sent to the student. This resolution is then added to the cache and knowledge base to train the system.


# Frontend Plan
Frontend Plan
Based on the provided images and the project requirements, here is a detailed frontend plan for the Masai LMS Support System, focusing on a clean, intuitive, and consistent user experience.

1. General UI/UX Principles
Font: The user interface will use the "Inter" font, as seen in the reference images.
Color Palette: A professional and calm color scheme will be used, primarily featuring:
Primary: rgb(82, 92, 235) (a distinct purple/blue for buttons and highlights).
Secondary: rgb(241, 243, 245) (a light gray for backgrounds and card elements).
Text: rgb(33, 37, 41) (dark gray for primary text), with lighter shades for secondary text.
Status Indicators: Green for Resolved, orange for Work in Progress, red for Action Required.
Styling & Spacing:
Rounded Corners: All major UI elements, such as cards, buttons, and input fields, will have soft rounded corners to create a modern and friendly feel.
Card-based Layout: Content will be organized into distinct cards with shadows to give a sense of depth and separation.
Spacing: Consistent padding and margins will be applied (e.g., p-4 or m-4 in a Tailwind-like system) to ensure elements are not cluttered.

2. Screen-by-Screen Breakdown
The application will have a consistent layout with a persistent navigation bar on the left and a header bar at the top, as shown in the reference images.
2.1. Login Screen
Layout: A simple, centered card will contain the login form.
Fields: Two input fields for "Email" and "Password".
Button: A prominent "Login" button styled with the primary color (rgb(82, 92, 235)).
Aesthetics: The background should be a subtle gray (rgb(241, 243, 245)), with the login card having a white background and a light shadow.
2.2. Student & Admin Dashboard
Content: A placeholder dashboard will be displayed upon login. The dashboard will be a card-based layout, providing a high-level overview.
Navigation: The left sidebar will contain links to various sections, with "Support Ticket" being the most relevant for this project.
2.3. Student: Support Tickets List
Purpose: This screen displays all tickets submitted by the logged-in student. * Layout: The main content area will be a list of ticket cards.
Filters: Two tabs at the top, "Unresolved" and "Resolved", will allow students to filter their tickets.
Ticket Card: Each card will show:
Title: The ticket title.
Response Count: The number of responses.
Admin Info: The admin who last responded (or "Mac" as a placeholder for the bot).
Status: A colored badge indicating the ticket status (Resolved, Closed, Open, etc.).
Rating: If the ticket is resolved, a rating status will be displayed.
Action: A "Create Ticket" button in the top-right corner, styled in the primary color.
2.4. Student: Create Ticket Screen
Purpose: A form for students to submit a new support ticket. * Layout: A clean, single-page form within a main content card.
Fields:
Category: A dropdown menu to select the ticket category.
Title: A single-line input field.
Message: A rich-text editor for detailed messages, supporting basic formatting (bold, italic, lists).
Actions:
Attachments: An icon button to add attachments (using Cloudinary).
Submit: A "Create Ticket" button in the bottom-right corner.
2.5. Student: Ticket Detail Screen
Purpose: Displays the full conversation history and allows for further interaction. * Layout: A scrollable conversation thread.
Header: The ticket title and status (e.g., "Update in Resume - RESOLVED").
Conversation:
The initial query will be displayed first.
Subsequent responses from "Anamika Basu" (the bot or admin) will follow in a chronological thread.
Each message will show the sender's name/initials, role, message content, and timestamp.
Rating/Feedback: A section with emojis will allow students to rate the resolution. This rating will update the database.
Actions:
A "Reopen Ticket" button will be visible for resolved tickets.
A "Bookmark" icon to save the ticket.
2.6. Admin: Document Management Screen
Purpose: A screen for admins to manage documents for the knowledge base.
Layout: A simple list of documents categorized by knowledge base.
Actions:
Upload: A button to upload a new document, which will trigger the backend process to update MongoDB and Pinecone.
Replace: An option on each document card to replace the file, which again triggers the backend update.
Delete: An icon to delete a document, with a confirmation modal, which will remove the document from both MongoDB and Pinecone.

3. General Components
Header Bar: A consistent header with:
Logo.
A search bar.
Icons for notifications, settings, and the user profile image with a dropdown.
A notification icon that will have a badge count for new, unread notifications. Clicking it will display a list of notifications, and viewing them will clear the badge count.
Left Navigation Bar: A vertical bar with icons and labels for different sections. "Support Ticket" will be a prominent item.
Notifications: A non-intrusive modal or toast notification will appear in the top-right corner to alert logged-in admins of new assigned tickets.
Empty States: Screens like the ticket list will have clear messaging when there are no tickets to display, encouraging the user to take action.

