import os
import uuid
from typing import List, Dict, Any, Optional
from fastapi import UploadFile
import pandas as pd
from pymongo import MongoClient
from pinecone import Pinecone
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from backend.app.core.config import settings
from backend.app.db.base import get_mongodb
import logging
import tempfile
import mimetypes

logger = logging.getLogger(__name__)

class DocumentService:
    def __init__(self):
        self.mongodb = get_mongodb()
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-001",
            google_api_key=settings.GOOGLE_API_KEY
        )
        
        # Initialize Pinecone
        if settings.PINECONE_API_KEY:
            self.pinecone = Pinecone(api_key=settings.PINECONE_API_KEY)
            try:
                self.index = self.pinecone.Index(settings.PINECONE_INDEX_NAME)
            except Exception as e:
                logger.error(f"Failed to connect to Pinecone index: {str(e)}")
                self.index = None
        else:
            self.pinecone = None
            self.index = None
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
    
    async def upload_document(self, file: UploadFile, category: str) -> Dict[str, Any]:
        """Upload and process any type of document"""
        try:
            # Generate document ID
            doc_id = str(uuid.uuid4())
            
            # Read file content
            content = await file.read()
            
            # Get MIME type
            mime_type, _ = mimetypes.guess_type(file.filename)
            if mime_type is None:
                mime_type = file.content_type or 'application/octet-stream'
            
            # Process based on file type
            text_content = await self._extract_text_content(content, file.filename, mime_type)
            
            if not text_content or len(text_content.strip()) == 0:
                raise ValueError("No text content could be extracted from the file")
            
            # Split into chunks
            chunks = self.text_splitter.split_text(text_content)
            
            # Store in MongoDB
            document = {
                "doc_id": doc_id,
                "file_name": file.filename,
                "content": text_content,
                "category": category,
                "chunk_count": len(chunks),
                "metadata": {
                    "file_type": mime_type,
                    "file_size": len(content),
                    "original_filename": file.filename
                }
            }
             
            collection_name = category
            collection = self.mongodb[collection_name]
            collection.insert_one(document)
            
            # Store in Pinecone
            if self.index:
                await self._store_in_pinecone(doc_id, chunks, category, file.filename)
            
            logger.info(f"Document uploaded successfully: {file.filename} ({len(chunks)} chunks)")
            
            return {
                "document_id": doc_id,
                "chunks_created": len(chunks)
            }
            
        except Exception as e:
            logger.error(f"Document upload error: {str(e)}")
            raise e
    
    async def _extract_text_content(self, content: bytes, filename: str, mime_type: str) -> str:
        """Extract text content from various file types"""
        try:
            # Create temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(filename)[1]) as temp_file:
                temp_file.write(content)
                temp_path = temp_file.name
            
            try:
                text_content = ""
                
                # PDF files
                if mime_type == 'application/pdf' or filename.lower().endswith('.pdf'):
                    text_content = await self._extract_from_pdf(temp_path)
                
                # Excel files
                elif (mime_type in ['application/vnd.ms-excel', 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'] 
                      or filename.lower().endswith(('.xlsx', '.xls', '.csv'))):
                    text_content = await self._extract_from_excel(temp_path)
                
                # Word documents
                elif (mime_type in ['application/msword', 'application/vnd.openxmlformats-officedocument.wordprocessingml.document']
                      or filename.lower().endswith(('.doc', '.docx'))):
                    text_content = await self._extract_from_word(temp_path)
                
                # Text files
                elif mime_type.startswith('text/') or filename.lower().endswith(('.txt', '.md', '.py', '.js', '.html', '.css', '.json', '.xml', '.yaml', '.yml')):
                    text_content = await self._extract_from_text(temp_path)
                
                # PowerPoint files
                elif (mime_type in ['application/vnd.ms-powerpoint', 'application/vnd.openxmlformats-officedocument.presentationml.presentation']
                      or filename.lower().endswith(('.ppt', '.pptx'))):
                    text_content = await self._extract_from_powerpoint(temp_path)
                
                # Image files (OCR)
                elif mime_type.startswith('image/') or filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff')):
                    text_content = await self._extract_from_image(temp_path)
                
                # Try as text if unknown type
                else:
                    try:
                        text_content = content.decode('utf-8', errors='ignore')
                    except:
                        raise ValueError(f"Unsupported file type: {mime_type}")
                
                return text_content
                
            finally:
                # Clean up temp file
                try:
                    os.unlink(temp_path)
                except:
                    pass
                    
        except Exception as e:
            logger.error(f"Text extraction error for {filename}: {str(e)}")
            raise e
    
    async def _extract_from_pdf(self, file_path: str) -> str:
        """Extract text from PDF files"""
        try:
            from pypdf import PdfReader
            with open(file_path, 'rb') as file:
                reader = PdfReader(file)
                text_content = []
                for page_num, page in enumerate(reader.pages):
                    page_text = page.extract_text()
                    if page_text.strip():
                        text_content.append(f"Page {page_num + 1}:\n{page_text}")
                
                return "\n\n".join(text_content)
        except Exception as e:
            logger.error(f"PDF extraction error: {str(e)}")
            # Fallback to langchain PDF loader
            try:
                loader = PyPDFLoader(file_path)
                documents = loader.load()
                return "\n\n".join([doc.page_content for doc in documents])
            except Exception as e2:
                logger.error(f"Fallback PDF extraction error: {str(e2)}")
                raise e
    
    async def _extract_from_excel(self, file_path: str) -> str:
        """Extract text from Excel files"""
        try:
            # Handle CSV files
            if file_path.lower().endswith('.csv'):
                df = pd.read_csv(file_path)
                return df.to_string(index=False)
            
            # Handle Excel files
            df_dict = pd.read_excel(file_path, sheet_name=None)
            text_content = []
            
            for sheet_name, df in df_dict.items():
                text_content.append(f"Sheet: {sheet_name}")
                text_content.append(df.to_string(index=False))
                text_content.append("\n" + "="*50 + "\n")
            
            return "\n".join(text_content)
            
        except Exception as e:
            logger.error(f"Excel extraction error: {str(e)}")
            raise e
    
    async def _extract_from_word(self, file_path: str) -> str:
        """Extract text from Word documents"""
        try:
            from docx import Document
            doc = Document(file_path)
            text_content = []
            
            for paragraph in doc.paragraphs:
                if paragraph.text.strip():
                    text_content.append(paragraph.text)
            
            # Extract tables
            for table in doc.tables:
                for row in table.rows:
                    row_text = [cell.text.strip() for cell in row.cells]
                    text_content.append(" | ".join(row_text))
            
            return "\n".join(text_content)
            
        except Exception as e:
            logger.error(f"Word extraction error: {str(e)}")
            # Fallback to docx2txt
            try:
                loader = Docx2txtLoader(file_path)
                documents = loader.load()
                return "\n\n".join([doc.page_content for doc in documents])
            except Exception as e2:
                logger.error(f"Fallback Word extraction error: {str(e2)}")
                raise e
    
    async def _extract_from_text(self, file_path: str) -> str:
        """Extract text from text files"""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
                return file.read()
        except UnicodeDecodeError:
            # Try different encodings
            for encoding in ['latin-1', 'cp1252', 'iso-8859-1']:
                try:
                    with open(file_path, 'r', encoding=encoding) as file:
                        return file.read()
                except:
                    continue
            raise ValueError("Could not decode text file")
    
    async def _extract_from_powerpoint(self, file_path: str) -> str:
        """Extract text from PowerPoint files"""
        try:
            from pptx import Presentation
            prs = Presentation(file_path)
            text_content = []
            
            for slide_num, slide in enumerate(prs.slides):
                slide_text = [f"Slide {slide_num + 1}:"]
                for shape in slide.shapes:
                    if hasattr(shape, "text") and shape.text.strip():
                        slide_text.append(shape.text)
                text_content.append("\n".join(slide_text))
            
            return "\n\n".join(text_content)
            
        except Exception as e:
            logger.error(f"PowerPoint extraction error: {str(e)}")
            raise e
    
    async def _extract_from_image(self, file_path: str) -> str:
        """Extract text from images using OCR"""
        try:
            # This requires pytesseract and PIL
            from PIL import Image
            import pytesseract
            
            image = Image.open(file_path)
            text = pytesseract.image_to_string(image)
            return text.strip()
            
        except ImportError:
            logger.warning("OCR dependencies not available (pytesseract, PIL)")
            return "Image file uploaded - OCR not available"
        except Exception as e:
            logger.error(f"Image OCR error: {str(e)}")
            return "Image file uploaded - OCR failed"
    
    async def _store_in_pinecone(self, doc_id: str, chunks: List[str], category: str, filename: str):
        """Store document chunks in Pinecone vector database"""
        try:
            if not self.index:
                logger.warning("Pinecone index not available, skipping vector storage")
                return
            
            vectors = []
            for i, chunk in enumerate(chunks):
                # Generate embedding
                embedding = await self.embeddings.aembed_query(chunk)
                
                # Create vector with metadata
                vector_id = f"{doc_id}_chunk_{i}"
                vector = {
                    "id": vector_id,
                    "values": embedding,
                    "metadata": {
                        "doc_id": doc_id,
                        "chunk_index": i,
                        "category": category,
                        "filename": filename,
                        "content": chunk[:500]  # Store first 500 chars for reference
                    }
                }
                vectors.append(vector)
            
            # Upsert vectors to Pinecone in batches
            batch_size = 100
            for i in range(0, len(vectors), batch_size):
                batch = vectors[i:i + batch_size]
                self.index.upsert(vectors=batch)
            
            logger.info(f"Stored {len(vectors)} vectors in Pinecone for document {doc_id}")
            
        except Exception as e:
            logger.error(f"Pinecone storage error: {str(e)}")
            raise e
    
    async def delete_document(self, doc_id: str):
        """Delete document from MongoDB and Pinecone"""
        try:
            # Find and delete from MongoDB
            deleted = False
            # Only check our 3 valid document collections
            valid_collections = ["program_details_documents", "qa_documents", "curriculum_documents"]
            
            for collection_name in valid_collections:
                if collection_name in self.mongodb.list_collection_names():
                    collection = self.mongodb[collection_name]
                    result = collection.delete_one({"doc_id": doc_id})
                    if result.deleted_count > 0:
                        deleted = True
                        break
            
            if not deleted:
                raise ValueError(f"Document {doc_id} not found")
            
            # Delete from Pinecone
            if self.index:
                # Find all vector IDs for this document
                query_response = self.index.query(
                    vector=[0] * 768,  # Dummy vector for metadata filtering
                    filter={"doc_id": doc_id},
                    top_k=1000,
                    include_metadata=True
                )
                
                vector_ids = [match.id for match in query_response.matches]
                if vector_ids:
                    self.index.delete(ids=vector_ids)
            
            logger.info(f"Document {doc_id} deleted successfully")
            
        except Exception as e:
            logger.error(f"Document deletion error: {str(e)}")
            raise e
    
    async def list_documents(self, category: Optional[str] = None) -> List[Dict[str, Any]]:
        """List documents in the knowledge base"""
        try:
            documents = []
            
            # Only check our 3 valid document collections
            valid_collections = ["program_details_documents", "qa_documents", "curriculum_documents"]
            
            for collection_name in valid_collections:
                if collection_name in self.mongodb.list_collection_names():
                    collection = self.mongodb[collection_name]
                    
                    query = {}
                    if category:
                        query["category"] = category
                    
                    for doc in collection.find(query, {"content": 0}):  # Exclude large content field
                        doc["_id"] = str(doc["_id"])  # Convert ObjectId to string
                        documents.append(doc)
            
            return documents
            
        except Exception as e:
            logger.error(f"Document listing error: {str(e)}")
            raise e