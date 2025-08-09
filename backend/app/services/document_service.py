#
# backend/app/services/document_service.py
#

import os
import uuid
import asyncio
import logging
import tempfile
import mimetypes
from typing import List, Dict, Any, Optional
import time
# Third-party imports
import pandas as pd
from fastapi import UploadFile
from pymongo import MongoClient
from pinecone import Pinecone, Index
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings

# Import parsers and handle potential ImportErrors
try:
    from pypdf import PdfReader
    from langchain_community.document_loaders import PyPDFLoader
except ImportError:
    PdfReader, PyPDFLoader = None, None

try:
    from docx import Document
    from langchain_community.document_loaders import Docx2txtLoader
except ImportError:
    Document, Docx2txtLoader = None, None

try:
    from pptx import Presentation
except ImportError:
    Presentation = None

try:
    from PIL import Image
    import pytesseract
except ImportError:
    Image, pytesseract = None, None

# Local application imports
# --- Assumed Configuration in backend/app/core/config.py ---
# from pydantic_settings import BaseSettings
#
# class Settings(BaseSettings):
#     # ... other settings
#     PINECONE_API_KEY: str
#     # New: A dictionary mapping categories to their Pinecone index names
#     # This should be populated from your .env file
#     PINECONE_INDEX_MAP: Dict[str, str] = {
#         "Program Details": "program-details-index",
#         "Q&A": "qa-index",
#         "Curriculum Documents": "curriculum-index"
#     }
#     # New: A dictionary mapping categories to their MongoDB collection names
#     MONGO_COLLECTION_MAP: Dict[str, str] = {
#         "Program Details": "program_details_documents",
#         "Q&A": "qa_documents",
#         "Curriculum Documents": "curriculum_documents"
#     }
#
# settings = Settings()
# -----------------------------------------------------------------
from backend.app.core.config import settings
from backend.app.db.base import get_mongodb

logger = logging.getLogger(__name__)


import functools

async def run_in_threadpool(func, *args, **kwargs):
    """Runs a synchronous function in a separate thread to avoid blocking."""
    loop = asyncio.get_running_loop()
    func = functools.partial(func, *args, **kwargs)
    return await loop.run_in_executor(None, func)


import gridfs

class DocumentService:
    """
    Manages document ingestion, storage, deletion, and searching across
    multiple dedicated Pinecone indices based on document category.
    """
    def __init__(self):
        self.mongodb = get_mongodb()
        self.gridfs = gridfs.GridFS(self.mongodb)
        self.embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2", 
        model_kwargs={"device": "cpu"}, 
        encode_kwargs={"normalize_embeddings": True}
        )

        # --- Refactor for Multi-Index Support ---
        self.pinecone_indices: Dict[str, Index] = {}
        if settings.PINECONE_API_KEY:
            self.pinecone = Pinecone(api_key=settings.PINECONE_API_KEY)
            # Initialize a Pinecone Index object for each configured index
            for category, index_name in settings.PINECONE_INDEX_MAP.items():
                try:
                    self.pinecone_indices[category] = self.pinecone.Index(index_name)
                    logger.info(f"Successfully connected to Pinecone index: '{index_name}' for category '{category}'")
                except Exception as e:
                    logger.error(f"Failed to connect to Pinecone index '{index_name}': {e}")
        else:
            self.pinecone = None
            logger.warning("PINECONE_API_KEY not set. Pinecone operations will be skipped.")
        # -------------------------------------------

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        self.collection_map = settings.MONGO_COLLECTION_MAP
        self.valid_categories = self.collection_map

    def _get_index(self, category: str) -> Optional[Index]:
        """Safely retrieves the Pinecone index for a given category."""
        index = self.pinecone_indices.get(category)
        if not index:
            logger.warning(f"No Pinecone index configured for category '{category}'. Skipping operation.")
        return index

    async def upload_document(self, file: UploadFile, category: str) -> Dict[str, Any]:
        """Upload, process, and index a document into its specified category index."""
        if category not in self.collection_map:
            raise ValueError(f"Invalid category '{category}'. Must be one of: {list(self.collection_map.keys())}")
        
        try:
            doc_id = str(uuid.uuid4())
            content = await file.read()
            mime_type, _ = mimetypes.guess_type(file.filename)
            mime_type = mime_type or file.content_type or 'application/octet-stream'

            text_content = await self._extract_text_content(content, file.filename, mime_type)
            if not text_content or not text_content.strip():
                raise ValueError("No text content could be extracted from the file.")

            chunks = await run_in_threadpool(self.text_splitter.split_text, text_content)

            # Store the large file content in GridFS
            gridfs_id = await run_in_threadpool(self.gridfs.put, content, filename=file.filename, content_type=mime_type)

            # Store metadata in the correct MongoDB collection (without the full content)
            collection_name = self.collection_map[category]
            document = {
                "doc_id": doc_id, "file_name": file.filename, "gridfs_id": str(gridfs_id),
                "category": category, "chunk_count": len(chunks),
                "metadata": {"file_type": mime_type, "file_size": len(content)}
            }
            collection = self.mongodb[collection_name]
            await run_in_threadpool(collection.insert_one, document)
            
            # Embed and store in the category-specific Pinecone index
            index = self._get_index(category)
            if index:
                await self._store_in_pinecone(index, doc_id, chunks, category, file.filename)
            
            logger.info(f"Document uploaded to category '{category}': {file.filename}")
            return {"document_id": doc_id, "chunks_created": len(chunks)}
            
        except Exception as e:
            print(f"Upload error for '{file.filename}': ",e)
            raise e

    async def _store_in_pinecone(self, index: Index, doc_id: str, chunks: List[str], category: str, filename: str):
        """Asynchronously embed and store document chunks in a specific Pinecone index."""
        try:
            retries = 3
            backoff_time = 1
            for i in range(retries):
                try:
                    embeddings = await self.embeddings.aembed_query(chunks)
                    break
                except Exception as e:
                    if "504 Deadline Exceeded" in str(e) and i < retries - 1:
                        print(f"Embedding failed: {e}. Retrying in {backoff_time} seconds...")
                        time.sleep(backoff_time)
                        backoff_time *= 2  # Exponential backoff
                    else:
                        print(f"Embedding failed after {i + 1} attempts: {e}")
                        raise # Re-raise the exception if all retries fail
            vectors = [
                {"id": f"{doc_id}_chunk_{i}", "values": embedding,
                 "metadata": {"doc_id": doc_id, "category": category, "filename": filename, "text_snippet": chunk[:500]}}
                for i, (chunk, embedding) in enumerate(zip(chunks, embeddings))
            ]
            
            # Upsert in batches
            for i in range(0, len(vectors), 100):
                batch = vectors[i:i + 100]
                await run_in_threadpool(index.upsert, vectors=batch)

            index_name = next((name for name, idx in self.pinecone_indices.items() if idx == index), "unknown")
            logger.info(f"Stored {len(vectors)} vectors in Pinecone index '{index_name}' for doc {doc_id}")
        except Exception as e:
            logger.error(f"Pinecone storage error for index '{index.name}': {e}")
            raise e

    async def delete_document(self, doc_id: str):
        """Deletes a document from MongoDB and its vectors from the corresponding Pinecone index."""
        try:
            doc_category = None
            # Find the document in any of the Mongo collections to get its category
            for category, collection_name in self.collection_map.items():
                collection = self.mongodb[collection_name]
                doc = await run_in_threadpool(collection.find_one_and_delete, {"doc_id": doc_id})
                if doc and 'gridfs_id' in doc:
                    await run_in_threadpool(self.gridfs.delete, doc['gridfs_id'])
                    doc_category = category
                    logger.info(f"Deleted document {doc_id} and its GridFS file from MongoDB collection '{collection_name}'.")
                    break
            
            if not doc_category:
                raise ValueError(f"Document {doc_id} not found in any collection.")

            # Delete from the corresponding Pinecone index
            index = self._get_index(doc_category)
            if index:
                await run_in_threadpool(index.delete, filter={"doc_id": doc_id})
                logger.info(f"Deleted vectors for {doc_id} from Pinecone index '{index.name}'.")

        except Exception as e:
            logger.error(f"Deletion error for doc {doc_id}: {e}")
            raise e

    # --- New Search Functionality ---
    async def search_documents(self, query: str, categories: Optional[List[str]] = None, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for documents by embedding a query and searching one or more Pinecone indices.

        Args:
            query: The user's search query.
            categories: A list of categories to search. If None, searches all configured indices.
            top_k: The number of results to return.

        Returns:
            A list of ranked search results.
        """
        if not self.pinecone:
            logger.warning("Pinecone is not configured. Cannot perform search.")
            return []
        
        print("--- DOCUMENT SERVICE DEBUG ---")
        print(f"Received categories for search: {categories}")
        print(f"Available Pinecone index keys in service: {list(self.pinecone_indices.keys())}")

        # Determine which indices to search
        indices_to_search = {}
        if categories:
            for cat in categories:
                if cat in self.pinecone_indices:
                    indices_to_search[cat] = self.pinecone_indices[cat]
        else: # If no categories are specified, search all of them
            indices_to_search = self.pinecone_indices

        if not indices_to_search:
            logger.warning(f"No valid indices found for categories: {categories}")
            return []

        try:
            # 1. Generate a single embedding for the query
            retries = 3
            backoff_time = 1
            for i in range(retries):
                try:
                    query_embedding = await self.embeddings.aembed_query(query)
                    break
                except Exception as e:
                    if "504 Deadline Exceeded" in str(e) and i < retries - 1:
                        print(f"Embedding failed: {e}. Retrying in {backoff_time} seconds...")
                        time.sleep(backoff_time)
                        backoff_time *= 2  # Exponential backoff
                    else:
                        print(f"Embedding failed after {i + 1} attempts: {e}")
                        raise # Re-raise the exception if all retries fail
            print(f"Generated embedding vector of length: {len(query_embedding)} in document service")

            # 2. Asynchronously query all specified indices in parallel
            async def query_index(index: Index):
                return await run_in_threadpool(
                    index.query,
                    vector=query_embedding,
                    top_k=top_k,
                    include_metadata=True
                )

            tasks = [query_index(index) for index in indices_to_search.values()]
            query_results = await asyncio.gather(*tasks)

            # 3. Merge, rank, and format the results
            all_matches = []
            for result in query_results:
                all_matches.extend(result.get('matches', []))

            # Sort all collected matches by their similarity score in descending order
            sorted_matches = sorted(all_matches, key=lambda x: x.get('score', 0), reverse=True)
            
            # 4. Format the final output
            final_results = []
            for match in sorted_matches[:top_k]:
                metadata = match.get('metadata', {})
                final_results.append({
                    "score": match.get('score'),
                    "category": metadata.get('category'),
                    "filename": metadata.get('filename'),
                    "doc_id": metadata.get('doc_id'),
                    "text_snippet": metadata.get('text_snippet')
                })
            
            return final_results

        except Exception as e:
            logger.error(f"Error during document search for query '{query}': {e}")
            raise e

    async def _extract_text_content(self, content: bytes, filename: str, mime_type: str) -> str:
        """Dispatcher to extract text from various file types using non-blocking calls."""
        suffix = os.path.splitext(filename)[1].lower()
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            temp_file.write(content)
            temp_path = temp_file.name
        
        try:
            # PDF files
            if mime_type == 'application/pdf' or suffix == '.pdf':
                return await self._extract_from_pdf(temp_path)
            # Excel & CSV files
            elif suffix in ['.xlsx', '.xls', '.csv']:
                return await self._extract_from_excel(temp_path)
            # Word documents
            elif suffix in ['.doc', '.docx']:
                 return await self._extract_from_word(temp_path)
            # PowerPoint files
            elif suffix in ['.ppt', '.pptx']:
                return await self._extract_from_powerpoint(temp_path)
            # Image files (OCR)
            elif mime_type.startswith('image/'):
                return await self._extract_from_image(temp_path)
            # Text files (fallback for many types)
            else:
                return await self._extract_from_text(content)
        finally:
            # Clean up temp file
            await run_in_threadpool(os.unlink, temp_path)

    async def _extract_from_pdf(self, file_path: str) -> str:
        if not PdfReader:
            raise ImportError("PDF processing libraries not installed. Please run 'pip install pypdf langchain-community'.")
        try:
            text_content = []
            reader = await run_in_threadpool(PdfReader, file_path)
            for i, page in enumerate(reader.pages):
                page_text = await run_in_threadpool(page.extract_text)
                if page_text and page_text.strip():
                    text_content.append(f"--- Page {i+1} ---\n{page_text.strip()}")
            return "\n\n".join(text_content)
        except Exception as e:
            logger.warning(f"pypdf failed for {file_path}: {e}. Falling back to Langchain loader.")
            try:
                loader = PyPDFLoader(file_path)
                docs = await run_in_threadpool(loader.load)
                return "\n\n".join([doc.page_content for doc in docs])
            except Exception as e2:
                logger.error(f"Fallback PDF loader also failed: {e2}")
                raise e2

    async def _extract_from_excel(self, file_path: str) -> str:
        try:
            if file_path.endswith('.csv'):
                df = await run_in_threadpool(pd.read_csv, file_path)
                return await run_in_threadpool(df.to_string, index=False)
            else:
                # Reading all sheets
                all_sheets = await run_in_threadpool(pd.read_excel, file_path, sheet_name=None)
                text_content = []
                for sheet_name, df in all_sheets.items():
                    text_content.append(f"--- Sheet: {sheet_name} ---\n{df.to_string(index=False)}")
                return "\n\n".join(text_content)
        except Exception as e:
            logger.error(f"Excel extraction error: {e}")
            raise e

    async def _extract_from_word(self, file_path: str) -> str:
        if not Document:
            raise ImportError("Word processing library not installed. Please run 'pip install python-docx'.")
        try:
            doc = await run_in_threadpool(Document, file_path)
            text_parts = [p.text for p in doc.paragraphs if p.text and p.text.strip()]
            return "\n".join(text_parts)
        except Exception as e:
            logger.warning(f"python-docx failed: {e}. Falling back to Docx2txtLoader.")
            try:
                loader = Docx2txtLoader(file_path)
                docs = await run_in_threadpool(loader.load)
                return "\n\n".join([doc.page_content for doc in docs])
            except Exception as e2:
                logger.error(f"Fallback Word loader failed: {e2}")
                raise e2

    async def _extract_from_powerpoint(self, file_path: str) -> str:
        if not Presentation:
            raise ImportError("PowerPoint library not installed. Please run 'pip install python-pptx'.")
        try:
            prs = await run_in_threadpool(Presentation, file_path)
            text_content = []
            for i, slide in enumerate(prs.slides):
                slide_texts = [shape.text for shape in slide.shapes if hasattr(shape, "text") and shape.text.strip()]
                if slide_texts:
                    text_content.append(f"--- Slide {i+1} ---\n" + "\n".join(slide_texts))
            return "\n\n".join(text_content)
        except Exception as e:
            logger.error(f"PowerPoint extraction error: {e}")
            raise e
    
    async def _extract_from_image(self, file_path: str) -> str:
        if not Image or not pytesseract:
            logger.warning("OCR dependencies not installed ('pip install pytesseract pillow'). OCR is disabled.")
            raise ImportError("OCR dependencies not available.")
        try:
            return await run_in_threadpool(pytesseract.image_to_string, Image.open(file_path))
        except Exception as e:
            logger.error(f"Image OCR error: {e}. Ensure Tesseract-OCR is installed on the system.")
            raise e

    async def _extract_from_text(self, content: bytes) -> str:
        """Tries to decode text content with common encodings."""
        for encoding in ['utf-8', 'latin-1', 'cp1252']:
            try:
                return content.decode(encoding)
            except UnicodeDecodeError:
                continue
        raise ValueError("Could not decode text file with common encodings.")

    
            
    async def delete_document(self, doc_id: str):
        """Deletes a document from MongoDB and its associated vectors from Pinecone."""
        try:
            # 1. Delete from MongoDB
            deleted_in_mongo = False
            for collection_name in self.valid_categories.values():
                collection = self.mongodb[collection_name]
                result = await run_in_threadpool(collection.delete_one, {"doc_id": doc_id})
                if result.deleted_count > 0:
                    deleted_in_mongo = True
                    break
            
            if not deleted_in_mongo:
                raise ValueError(f"Document {doc_id} not found in MongoDB.")

            # 2. Delete from Pinecone using a metadata filter (more efficient)
            if self.index:
                await run_in_threadpool(self.index.delete, filter={"doc_id": doc_id})

            logger.info(f"Document {doc_id} and its vectors deleted successfully.")

        except Exception as e:
            logger.error(f"Document deletion error for {doc_id}: {e}")
            raise e

    async def list_documents(self, category: Optional[str] = None) -> List[Dict[str, Any]]:
        """Lists document metadata from MongoDB."""
        documents = []
        collections_to_search = self.valid_categories.values()

        if category:
            if category not in self.valid_categories:
                return []
            collections_to_search = [self.valid_categories[category]]

        for collection_name in collections_to_search:
            collection = self.mongodb[collection_name]
            # Use run_in_threadpool for the blocking cursor iteration
            async for doc in run_in_threadpool(collection.find, {}, {"content": 0}):
                doc["_id"] = str(doc["_id"])
                documents.append(doc)
        
        return documents
