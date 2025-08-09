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
from bson.objectid import ObjectId
import gridfs
import functools

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
from backend.app.core.config import settings
from backend.app.db.base import get_mongodb

logger = logging.getLogger(__name__)


async def run_in_threadpool(func, *args, **kwargs):
    """Runs a synchronous function in a separate thread to avoid blocking."""
    loop = asyncio.get_running_loop()
    # Use functools.partial to package the function and its arguments
    func_with_args = functools.partial(func, *args, **kwargs)
    return await loop.run_in_executor(None, func_with_args)


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

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        self.collection_map = settings.MONGO_COLLECTION_MAP
        self.valid_categories = self.collection_map.keys()

    def _get_index(self, category: str) -> Optional[Index]:
        """Safely retrieves the Pinecone index for a given category."""
        index = self.pinecone_indices.get(category)
        if not index:
            logger.warning(f"No Pinecone index configured for category '{category}'. Skipping operation.")
        return index

    async def _process_and_store_excel(self, file_content: bytes, filename: str, doc_id: str, category: str) -> Dict[str, Any]:
        """
        Processes a Q&A Excel file row by row, embedding the 'message'
        and storing the 'Potential response' in the vector's metadata.
        """
        print("--- Processing file using dedicated Excel Q&A logic ---")
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx") as temp_file:
                temp_file.write(file_content)
                temp_path = temp_file.name
            
            df = await run_in_threadpool(pd.read_excel, temp_path)
            os.unlink(temp_path)

            if "message" not in df.columns or "Potential response" not in df.columns:
                raise ValueError("Excel file must contain 'message' and 'Potential response' columns.")
            
            df = df[["message", "Potential response"]].dropna().reset_index()
            if df.empty:
                raise ValueError("No valid rows found in the Excel file.")

            messages_to_embed = df["message"].tolist()
            embeddings = await self.embeddings.aembed_documents(messages_to_embed)

            vectors = []
            for index, row in df.iterrows():
                vector = {
                    "id": f"{doc_id}_row_{index}",
                    "values": embeddings[index],
                    "metadata": {
                        "doc_id": doc_id,
                        "category": category,
                        "filename": filename,
                        "text_snippet": row["message"],          # The question
                        "potential_response": row["Potential response"]  # The answer
                    }
                }
                vectors.append(vector)

            index = self._get_index(category)
            if index:
                for i in range(0, len(vectors), 100):
                    batch = vectors[i:i + 100]
                    await run_in_threadpool(index.upsert, vectors=batch)
                logger.info(f"Stored {len(vectors)} Q&A pairs in Pinecone for doc {doc_id}")

            mime_type, _ = mimetypes.guess_type(filename)
            gridfs_id = await run_in_threadpool(self.gridfs.put, file_content, filename=filename, content_type=mime_type)

            collection_name = self.collection_map[category]
            document_metadata = {
                "doc_id": doc_id,
                "file_name": filename,
                "gridfs_id": str(gridfs_id),
                "category": category,
                "chunk_count": len(vectors),
                "metadata": {"file_type": mime_type, "file_size": len(file_content)}
            }
            collection = self.mongodb[collection_name]
            await run_in_threadpool(collection.insert_one, document_metadata)
            logger.info(f"Stored metadata for Excel doc {doc_id} in MongoDB collection '{collection_name}'.")
            
            return {"document_id": doc_id, "items_created": len(vectors)}

        except Exception as e:
            logger.error(f"Excel processing error for '{filename}': {e}", exc_info=True)
            raise e

    async def upload_document(self, file: UploadFile, category: str) -> Dict[str, Any]:
        """Upload, process, and index a document into its specified category index."""
        if category not in self.collection_map:
            raise ValueError(f"Invalid category '{category}'. Must be one of: {list(self.collection_map.keys())}")
        
        try:
            doc_id = str(uuid.uuid4())
            content = await file.read()

            mime_type, _ = mimetypes.guess_type(file.filename)
            is_excel = mime_type in [
                'application/vnd.ms-excel',
                'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
            ] or file.filename.endswith(('.xlsx', '.xls'))

            if is_excel:
                return await self._process_and_store_excel(content, file.filename, doc_id, category)

            final_mime_type = mime_type or file.content_type or 'application/octet-stream'
            text_content = await self._extract_text_content(content, file.filename, final_mime_type)
            if not text_content or not text_content.strip():
                raise ValueError("No text content could be extracted from the file.")

            chunks = self.text_splitter.split_text(text_content)

            gridfs_id = await run_in_threadpool(self.gridfs.put, content, filename=file.filename, content_type=final_mime_type)

            collection_name = self.collection_map[category]
            document = {
                "doc_id": doc_id, "file_name": file.filename, "gridfs_id": str(gridfs_id),
                "category": category, "chunk_count": len(chunks),
                "metadata": {"file_type": final_mime_type, "file_size": len(content)}
            }
            collection = self.mongodb[collection_name]
            await run_in_threadpool(collection.insert_one, document)
            
            index = self._get_index(category)
            if index:
                await self._store_in_pinecone(index, doc_id, chunks, category, file.filename)
            
            logger.info(f"Document uploaded to category '{category}': {file.filename}")
            return {"document_id": doc_id, "items_created": len(chunks)}
            
        except Exception as e:
            logger.error(f"Upload error for '{file.filename}': {e}", exc_info=True)
            raise e
            
    async def _store_in_pinecone(self, index: Index, doc_id: str, chunks: List[str], category: str, filename: str):
        """Asynchronously embed and store document chunks in a specific Pinecone index."""
        try:
            # 1. Correctly use embed_documents for a list of chunks and wrap in threadpool
            embeddings = await run_in_threadpool(self.embeddings.embed_documents, chunks)

            vectors = [
                {"id": f"{doc_id}_chunk_{i}", "values": embedding,
                "metadata": {"doc_id": doc_id, "category": category, "filename": filename, "text_snippet": chunk[:500]}}
                for i, (chunk, embedding) in enumerate(zip(chunks, embeddings))
            ]
            
            # 2. Upsert in batches
            for i in range(0, len(vectors), 100):
                batch = vectors[i:i + 100]
                await run_in_threadpool(index.upsert, vectors=batch)

            index_name = settings.PINECONE_INDEX_MAP.get(category, "unknown")
            logger.info(f"Stored {len(vectors)} vectors in Pinecone index '{index_name}' for doc {doc_id}")

        except Exception as e:
            # 3. Correctly get the index name for logging from your settings map
            index_name = settings.PINECONE_INDEX_MAP.get(category, "unknown")
            logger.error(f"Pinecone storage error for index '{index_name}': {e}", exc_info=True)
            raise e
            

    async def delete_document(self, doc_id: str):
        """Deletes a document from MongoDB and its vectors from the corresponding Pinecone index."""
        try:
            doc_category = None
            found_doc = None

            for category, collection_name in self.collection_map.items():
                collection = self.mongodb[collection_name]
                doc = await run_in_threadpool(collection.find_one_and_delete, {"doc_id": doc_id})
                if doc:
                    found_doc = doc
                    doc_category = category
                    logger.info(f"Deleted document metadata {doc_id} from MongoDB collection '{collection_name}'.")
                    break
            
            if not found_doc:
                raise ValueError(f"Document {doc_id} not found in any collection.")

            if 'gridfs_id' in found_doc:
                try:
                    gridfs_id_obj = ObjectId(found_doc['gridfs_id'])
                    await run_in_threadpool(self.gridfs.delete, gridfs_id_obj)
                    logger.info(f"Deleted GridFS file for doc {doc_id}.")
                except Exception as e:
                    logger.error(f"Failed to delete GridFS file for doc {doc_id} with gridfs_id {found_doc['gridfs_id']}: {e}")

            index = self._get_index(doc_category)
            if index:
                await run_in_threadpool(index.delete, filter={"doc_id": doc_id})
                logger.info(f"Deleted vectors for {doc_id} from Pinecone index '{index.name}'.")

        except Exception as e:
            logger.error(f"Deletion error for doc {doc_id}: {e}", exc_info=True)
            raise e

    async def search_documents(self, query: str, categories: Optional[List[str]] = None, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search for documents by embedding a query and searching one or more Pinecone indices."""
        if not self.pinecone:
            logger.warning("Pinecone is not configured. Cannot perform search.")
            return []
        
        indices_to_search = {}
        target_categories = categories or self.pinecone_indices.keys()
        for cat in target_categories:
            if cat in self.pinecone_indices:
                indices_to_search[cat] = self.pinecone_indices[cat]

        if not indices_to_search:
            logger.warning(f"No valid indices found for specified categories: {categories}")
            return []

        try:
            query_embedding = await self.embeddings.aembed_query(query)

            async def query_index(index: Index):
                return await run_in_threadpool(
                    index.query,
                    vector=query_embedding,
                    top_k=top_k,
                    include_metadata=True
                )

            tasks = [query_index(index) for index in indices_to_search.values()]
            query_results = await asyncio.gather(*tasks)

            all_matches = []
            for result in query_results:
                all_matches.extend(result.get('matches', []))

            sorted_matches = sorted(all_matches, key=lambda x: x.get('score', 0), reverse=True)
            
            final_results = []
            for match in sorted_matches[:top_k]:
                metadata = match.get('metadata', {})
                final_results.append({
                    "score": match.get('score'),
                    "category": metadata.get('category'),
                    "filename": metadata.get('filename'),
                    "doc_id": metadata.get('doc_id'),
                    "text_snippet": metadata.get('text_snippet'),
                    "potential_response": metadata.get('potential_response')
                })
            
            return final_results

        except Exception as e:
            logger.error(f"Error during document search for query '{query}': {e}", exc_info=True)
            raise e

    async def _extract_text_content(self, content: bytes, filename: str, mime_type: str) -> str:
        """Dispatcher to extract text from various file types using non-blocking calls."""
        suffix = os.path.splitext(filename)[1].lower()
        
        # Use a temporary file for libraries that need a file path
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            temp_file.write(content)
            temp_path = temp_file.name
        
        try:
            if mime_type == 'application/pdf' or suffix == '.pdf':
                return await self._extract_from_pdf(temp_path)
            elif suffix in ['.doc', '.docx']:
                return await self._extract_from_word(temp_path)
            elif suffix in ['.ppt', '.pptx']:
                return await self._extract_from_powerpoint(temp_path)
            elif mime_type.startswith('image/'):
                return await self._extract_from_image(temp_path)
            else: # Fallback for text-based files like .txt, .csv, etc.
                return await self._extract_from_text(content)
        finally:
            os.unlink(temp_path)

    async def _extract_from_pdf(self, file_path: str) -> str:
        if not PdfReader:
            raise ImportError("PDF processing library not installed. Run 'pip install pypdf'.")
        try:
            reader = await run_in_threadpool(PdfReader, file_path)
            text_content = []
            for i, page in enumerate(reader.pages):
                page_text = page.extract_text()
                if page_text and page_text.strip():
                    text_content.append(f"--- Page {i+1} ---\n{page_text.strip()}")
            return "\n\n".join(text_content)
        except Exception as e:
            logger.warning(f"pypdf failed for {file_path}: {e}. Falling back to Langchain loader.")
            if not PyPDFLoader:
                raise ImportError("Langchain PDF loader not installed. Run 'pip install langchain-community'.")
            try:
                loader = PyPDFLoader(file_path)
                docs = await run_in_threadpool(loader.load)
                return "\n\n".join([doc.page_content for doc in docs])
            except Exception as e2:
                logger.error(f"Fallback PDF loader also failed: {e2}")
                raise e2

    async def _extract_from_word(self, file_path: str) -> str:
        if not Document:
            raise ImportError("Word processing library not installed. Run 'pip install python-docx'.")
        try:
            doc = await run_in_threadpool(Document, file_path)
            return "\n".join([p.text for p in doc.paragraphs if p.text and p.text.strip()])
        except Exception as e:
            logger.warning(f"python-docx failed: {e}. Falling back to Docx2txtLoader.")
            if not Docx2txtLoader:
                raise ImportError("Langchain Word loader not installed. Run 'pip install docx2txt'.")
            try:
                loader = Docx2txtLoader(file_path)
                docs = await run_in_threadpool(loader.load)
                return "\n\n".join([doc.page_content for doc in docs])
            except Exception as e2:
                logger.error(f"Fallback Word loader failed: {e2}")
                raise e2

    async def _extract_from_powerpoint(self, file_path: str) -> str:
        if not Presentation:
            raise ImportError("PowerPoint library not installed. Run 'pip install python-pptx'.")
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
            raise ImportError("OCR dependencies not installed. Run 'pip install pytesseract pillow'.")
        try:
            # Pytesseract is CPU-bound, so run it in a thread
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

    async def list_documents(self, category: Optional[str] = None) -> List[Dict[str, Any]]:
        """Lists document metadata from MongoDB, handling blocking calls correctly."""
        documents = []
        collections_to_search = []
        if category:
            if category in self.collection_map:
                collections_to_search.append(self.collection_map[category])
        else:
            collections_to_search = self.collection_map.values()
        
        if not collections_to_search:
            return []

        def fetch_and_process_docs(collection):
            """Sync function to be run in a thread."""
            docs = []
            for doc in collection.find({}):
                doc["_id"] = str(doc["_id"]) # Convert ObjectId to string
                docs.append(doc)
            return docs

        for collection_name in collections_to_search:
            collection = self.mongodb[collection_name]
            # CORRECTED: Run the entire blocking DB operation in the thread pool
            docs_list = await run_in_threadpool(fetch_and_process_docs, collection)
            documents.extend(docs_list)
        
        return documents