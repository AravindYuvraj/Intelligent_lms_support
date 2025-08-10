# backend/app/services/document_service.py

import os
import uuid
import asyncio
import logging
import tempfile
import mimetypes
from typing import List, Dict, Any, Optional
import io # NEW: Import io for handling bytes with pandas

# Third-party imports
import pandas as pd
from fastapi import UploadFile
from pymongo import MongoClient
from pinecone import Pinecone, Index
from bson.objectid import ObjectId
import gridfs
import functools

# Unstructured.io for unified document processing
from unstructured.partition.auto import partition
from unstructured.chunking.title import chunk_by_title

# Langchain and other imports
from langchain_huggingface import HuggingFaceEmbeddings
# NEW: Re-added Text Splitter for CSV and fallback processing
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Local application imports
from backend.app.core.config import settings
from backend.app.db.base import get_mongodb

logger = logging.getLogger(__name__)


async def run_in_threadpool(func, *args, **kwargs):
    """Runs a synchronous function in a separate thread to avoid blocking."""
    loop = asyncio.get_running_loop()
    func_with_args = functools.partial(func, *args, **kwargs)
    return await loop.run_in_executor(None, func_with_args)


class DocumentService:
    """
    Manages document ingestion, storage, deletion, and searching across
    multiple dedicated Pinecone indices based on document category, using
    unstructured.io for document processing.
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
            for category, index_name in settings.PINECONE_INDEX_MAP.items():
                try:
                    self.pinecone_indices[category] = self.pinecone.Index(index_name)
                    print(f"Successfully connected to Pinecone index: '{index_name}' for category '{category}'")
                except Exception as e:
                    print(f"Failed to connect to Pinecone index '{index_name}': {e}")
        else:
            self.pinecone = None
            print("PINECONE_API_KEY not set. Pinecone operations will be skipped.")

        # NEW: Initialize the text splitter for use with CSVs
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
            print(f"No Pinecone index configured for category '{category}'. Skipping operation.")
        return index

    async def _process_and_store_excel_qa(self, file_content: bytes, filename: str, doc_id: str, category: str) -> Dict[str, Any]:
        """
        Processes a Q&A Excel file row by row, embedding the 'message'
        and storing the 'Potential response' in the vector's metadata.
        This specialized logic is kept as requested.
        """
        # This function remains unchanged.
        print("--- Processing file using dedicated Excel Q&A logic ---")
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx") as temp_file:
                temp_file.write(file_content)
                temp_path = temp_file.name

            df = await run_in_threadpool(pd.read_excel, temp_path)
            os.unlink(temp_path)

            if "message" not in df.columns or "Potential response" not in df.columns:
                raise ValueError("Excel file must contain 'message' and 'Potential response' columns for Q&A processing.")

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
                        "text_snippet": row["message"],
                        "potential_response": row["Potential response"]
                    }
                }
                vectors.append(vector)

            pinecone_index = self._get_index(category)
            if pinecone_index:
                await run_in_threadpool(pinecone_index.upsert, vectors=vectors, batch_size=100)
                print(f"Stored {len(vectors)} Q&A pairs in Pinecone for doc {doc_id}")

            mime_type, _ = mimetypes.guess_type(filename)
            gridfs_id = await run_in_threadpool(self.gridfs.put, file_content, filename=filename, content_type=mime_type)

            collection_name = self.collection_map[category]
            document_metadata = {
                "doc_id": doc_id, "file_name": filename, "gridfs_id": str(gridfs_id),
                "category": category, "chunk_count": len(vectors),
                "metadata": {"file_type": mime_type, "file_size": len(file_content)}
            }
            collection = self.mongodb[collection_name]
            await run_in_threadpool(collection.insert_one, document_metadata)
            print(f"Stored metadata for Excel doc {doc_id} in MongoDB collection '{collection_name}'.")

            return {"document_id": doc_id, "items_created": len(vectors)}

        except Exception as e:
            print(f"Excel Q&A processing error for '{filename}': {e}")
            raise e

    async def upload_document(self, file: UploadFile, category: str) -> Dict[str, Any]:
        """Upload, process, and index a document into its specified category index."""
        if category not in self.collection_map:
            raise ValueError(f"Invalid category '{category}'. Must be one of: {list(self.collection_map.keys())}")

        try:
            doc_id = str(uuid.uuid4())
            content = await file.read()
            filename = file.filename
            chunks = []
            chunk_texts = []
            metadata_list = None

            # --- Start of Processing Logic ---

            is_excel_qa = False
            if filename.endswith(('.xlsx', '.xls')):
                try:
                    # Check if it matches the special Q&A format without raising an error
                    df_peek = pd.read_excel(io.BytesIO(content))
                    if "message" in df_peek.columns and "Potential response" in df_peek.columns:
                        is_excel_qa = True
                except Exception:
                    pass # Not a valid Excel file, will be handled by unstructured

            if is_excel_qa:
                return await self._process_and_store_excel_qa(content, filename, doc_id, category)

            # NEW: Add a dedicated path for CSV files to handle encoding errors robustly
            elif filename.endswith('.csv'):
                print(f"Processing '{filename}' as a CSV file.")
                try:
                    # Use pandas to read the CSV, trying different encodings
                    df = pd.read_csv(io.BytesIO(content))
                except UnicodeDecodeError:
                    print("Default CSV decoding failed, trying UTF-8.")
                    df = pd.read_csv(io.BytesIO(content), encoding='utf-8')

                # Convert the entire dataframe to a single string and then chunk it
                text_content = df.to_string(index=False)
                chunks = self.text_splitter.split_text(text_content)
                chunk_texts = chunks # For CSV, chunks and chunk_texts are the same

            # Fallback to unstructured.io for all other document types
            else:
                print(f"Processing '{filename}' with unstructured.io.")
                with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(filename)[1]) as temp_file:
                    temp_file.write(content)
                    temp_path = temp_file.name
                try:
                    elements = await run_in_threadpool(partition, filename=temp_path, content_type=file.content_type, strategy="fast")
                    chunk_elements = await run_in_threadpool(chunk_by_title, elements, max_characters=1000, new_after_n_chars=800)
                    chunk_texts = [c.text for c in chunk_elements]
                    metadata_list = [c.metadata.to_dict() for c in chunk_elements]
                    chunks = chunk_texts # Use the text content for the chunk count
                finally:
                    os.unlink(temp_path)

            if not chunks:
                raise ValueError("No text content could be extracted from the file.")

            # --- End of Processing Logic ---

            final_mime_type = file.content_type or 'application/octet-stream'
            gridfs_id = await run_in_threadpool(self.gridfs.put, content, filename=filename, content_type=final_mime_type)

            collection_name = self.collection_map[category]
            document = {
                "doc_id": doc_id, "file_name": filename, "gridfs_id": str(gridfs_id),
                "category": category, "chunk_count": len(chunks),
                "metadata": {"file_type": final_mime_type, "file_size": len(content)}
            }
            collection = self.mongodb[collection_name]
            await run_in_threadpool(collection.insert_one, document)

            pinecone_index = self._get_index(category)
            if pinecone_index:
                await self._store_in_pinecone(pinecone_index, doc_id, chunk_texts, category, filename, metadata_list)

            print(f"Document uploaded to category '{category}': {filename}")
            return {"document_id": doc_id, "items_created": len(chunks)}

        except Exception as e:
            print(f"Upload error for '{file.filename}': {e}")
            raise e

    async def _store_in_pinecone(self, index: Index, doc_id: str, chunks: List[str], category: str, filename: str, metadata_list: Optional[List[Dict]] = None):
        """Asynchronously embed and store document chunks in a specific Pinecone index."""
        try:
            embeddings = await self.embeddings.aembed_documents(chunks)
            vectors = []
            for i, (chunk_text, embedding) in enumerate(zip(chunks, embeddings)):
                vector_metadata = {
                    "doc_id": doc_id,
                    "category": category,
                    "filename": filename,
                    "text_snippet": chunk_text[:1000],
                }

                if metadata_list and i < len(metadata_list):
                    unstructured_meta = metadata_list[i]
                    
                    page_number = unstructured_meta.get("page_number")
                    if page_number is not None:
                        vector_metadata["page_number"] = page_number

                    element_type = unstructured_meta.get("category")
                    if element_type is not None:
                        vector_metadata["element_type"] = element_type
                
                # --- START: DEBUGGING LINE ---
                # This will print the exact metadata being prepared for each chunk.
                # Look for 'element_type': None in your console output.
                print(f"DEBUG METADATA: {vector_metadata}")
                # --- END: DEBUGGING LINE ---

                vector = {
                    "id": f"{doc_id}_chunk_{i}",
                    "values": embedding,
                    "metadata": vector_metadata
                }
                vectors.append(vector)

            await run_in_threadpool(index.upsert, vectors=vectors, batch_size=100)

            index_name = settings.PINECONE_INDEX_MAP.get(category, "unknown")
            logger.info(f"Stored {len(vectors)} vectors in Pinecone index '{index_name}' for doc {doc_id}")

        except Exception as e:
            index_name = settings.PINECONE_INDEX_MAP.get(category, "unknown")
            logger.error(f"Pinecone storage error for index '{index_name}': {e}")
            raise e
           
    async def delete_document(self, doc_id: str):
        # This method remains unchanged.
        """Deletes a document from MongoDB and its vectors from the corresponding Pinecone index."""
        try:
            doc_category = None
            found_doc = None

            for category, collection_name in self.collection_map.items():
                collection = self.mongodb[collection_name]
                doc = await run_in_threadpool(collection.find_one_and_delete, {"doc_id": doc_id})
                if doc:
                    if 'gridfs_id' in doc:
                        gridfs_object_id = ObjectId(doc['gridfs_id'])
                        await run_in_threadpool(self.gridfs.delete, gridfs_object_id)
                    
                    doc_category = category
                    found_doc = doc
                    print(
                        f"Deleted document {doc_id} and its GridFS file from MongoDB collection '{collection_name}'."
                    )
                    break

            if not found_doc:
                raise ValueError(f"Document {doc_id} not found in any collection.")

            index = self._get_index(doc_category)
            if index:
                index_name = settings.PINECONE_INDEX_MAP[doc_category]
                await run_in_threadpool(index.delete, filter={"doc_id": doc_id})
                print(f"Deleted vectors for {doc_id} from Pinecone index '{index_name}'.")

        except Exception as e:
            print(f"Deletion error for doc {doc_id}: {e}")
            raise e


    async def search_documents(self, query: str, categories: Optional[List[str]] = None, top_k: int = 5) -> List[Dict[str, Any]]:
        # This method remains unchanged.
        """Search for documents by embedding a query and searching one or more Pinecone indices."""
        if not self.pinecone:
            print("Pinecone is not configured. Cannot perform search.")
            return []

        indices_to_search = {}
        target_categories = categories or self.pinecone_indices.keys()
        for cat in target_categories:
            if cat in self.pinecone_indices:
                indices_to_search[cat] = self.pinecone_indices[cat]

        if not indices_to_search:
            print(f"No valid indices found for specified categories: {categories}")
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
                    "potential_response": metadata.get('potential_response'),
                    "page_number": metadata.get('page_number'),
                    "element_type": metadata.get('element_type')
                })

            return final_results

        except Exception as e:
            print(f"Error during document search for query '{query}': {e}")
            raise e

    async def list_documents(self, category: Optional[str] = None) -> List[Dict[str, Any]]:
        # This method remains unchanged.
        """Lists document metadata from MongoDB."""
        documents = []
        collections_to_search = []
        if category:
            if category in self.collection_map:
                collections_to_search.append(self.collection_map[category])
        else:
            collections_to_search = self.collection_map.values()

        if not collections_to_search:
            return []

        async def fetch_all_docs(collection_name: str):
            collection = self.mongodb[collection_name]
            docs = []
            cursor = collection.find({}, {"content": 0})
            for doc in await run_in_threadpool(list, cursor):
                doc["_id"] = str(doc["_id"])
                docs.append(doc)
            return docs

        tasks = [fetch_all_docs(name) for name in collections_to_search]
        results = await asyncio.gather(*tasks)
        for doc_list in results:
            documents.extend(doc_list)

        return documents