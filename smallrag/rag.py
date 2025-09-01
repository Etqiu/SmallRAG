"""
Core RAG functionality for SmallRAG
"""

import os
from typing import List, Dict, Any, Optional
from pathlib import Path

import chromadb
from chromadb.config import Settings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from langchain.schema import Document

from .document_loader import DocumentLoader
from .utils import setup_logging

logger = setup_logging(__name__)


class SmallRAG:
    """
    A lightweight Retrieval-Augmented Generation system
    """
    
    def __init__(
        self,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        persist_directory: str = "./chroma_db",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        temperature: float = 0.7
    ):
        """
        Initialize the SmallRAG system
        
        Args:
            embedding_model: HuggingFace model name for embeddings
            persist_directory: Directory to persist ChromaDB
            chunk_size: Size of text chunks for splitting
            chunk_overlap: Overlap between chunks
            temperature: Temperature for LLM responses
        """
        self.embedding_model = embedding_model
        self.persist_directory = persist_directory
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.temperature = temperature
        
        # Initialize components
        self._setup_embeddings()
        self._setup_text_splitter()
        self._setup_vectorstore()
        self._setup_llm()
        self._setup_qa_chain()
        
        self.document_loader = DocumentLoader()
        
    def _setup_embeddings(self):
        """Initialize the embedding model"""
        try:
            self.embeddings = HuggingFaceEmbeddings(
                model_name=self.embedding_model,
                model_kwargs={'device': 'cpu'}
            )
            logger.info(f"Initialized embeddings with model: {self.embedding_model}")
        except Exception as e:
            logger.error(f"Failed to initialize embeddings: {e}")
            raise
    
    def _setup_text_splitter(self):
        """Initialize the text splitter"""
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
        )
        logger.info("Initialized text splitter")
    
    def _setup_vectorstore(self):
        """Initialize the vector store"""
        try:
            self.vectorstore = Chroma(
                embedding_function=self.embeddings,
                persist_directory=self.persist_directory
            )
            logger.info(f"Initialized vector store at: {self.persist_directory}")
        except Exception as e:
            logger.error(f"Failed to initialize vector store: {e}")
            raise
    
    def _setup_llm(self):
        """Initialize the language model"""
        try:
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                logger.warning("OPENAI_API_KEY not found. Using mock LLM.")
                self.llm = None
            else:
                self.llm = ChatOpenAI(
                    temperature=self.temperature,
                    model_name="gpt-3.5-turbo"
                )
                logger.info("Initialized OpenAI LLM")
        except Exception as e:
            logger.error(f"Failed to initialize LLM: {e}")
            self.llm = None
    
    def _setup_qa_chain(self):
        """Initialize the QA chain"""
        if self.llm:
            self.qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=self.vectorstore.as_retriever(
                    search_kwargs={"k": 3}
                )
            )
            logger.info("Initialized QA chain")
        else:
            self.qa_chain = None
            logger.warning("QA chain not initialized due to missing LLM")
    
    def add_documents(self, file_paths: List[str]) -> None:
        """
        Add documents to the vector store
        
        Args:
            file_paths: List of file paths to add
        """
        documents = []
        
        for file_path in file_paths:
            try:
                # Load document
                doc_text = self.document_loader.load_document(file_path)
                
                # Split into chunks
                chunks = self.text_splitter.split_text(doc_text)
                
                # Create Document objects
                for i, chunk in enumerate(chunks):
                    doc = Document(
                        page_content=chunk,
                        metadata={
                            "source": file_path,
                            "chunk_id": i,
                            "total_chunks": len(chunks)
                        }
                    )
                    documents.append(doc)
                
                logger.info(f"Processed {len(chunks)} chunks from {file_path}")
                
            except Exception as e:
                logger.error(f"Failed to process {file_path}: {e}")
                continue
        
        if documents:
            # Add to vector store
            self.vectorstore.add_documents(documents)
            self.vectorstore.persist()
            logger.info(f"Added {len(documents)} document chunks to vector store")
    
    def query(self, question: str) -> Dict[str, Any]:
        """
        Query the RAG system
        
        Args:
            question: The question to ask
            
        Returns:
            Dictionary containing answer and metadata
        """
        if not self.qa_chain:
            return {
                "answer": "LLM not available. Please check your OpenAI API key.",
                "sources": [],
                "error": "LLM not initialized"
            }
        
        try:
            # Get answer from QA chain
            result = self.qa_chain({"query": question})
            answer = result["result"]
            
            # Get relevant documents
            docs = self.vectorstore.similarity_search(question, k=3)
            sources = [doc.metadata.get("source", "Unknown") for doc in docs]
            
            return {
                "answer": answer,
                "sources": sources,
                "question": question
            }
            
        except Exception as e:
            logger.error(f"Error during query: {e}")
            return {
                "answer": f"Error processing query: {str(e)}",
                "sources": [],
                "error": str(e)
            }
    
    def similarity_search(self, query: str, k: int = 5) -> List[Document]:
        """
        Perform similarity search
        
        Args:
            query: Search query
            k: Number of results to return
            
        Returns:
            List of relevant documents
        """
        try:
            return self.vectorstore.similarity_search(query, k=k)
        except Exception as e:
            logger.error(f"Error during similarity search: {e}")
            return []
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the vector store collection
        
        Returns:
            Dictionary with collection statistics
        """
        try:
            collection = self.vectorstore._collection
            count = collection.count()
            
            return {
                "total_documents": count,
                "embedding_model": self.embedding_model,
                "persist_directory": self.persist_directory
            }
        except Exception as e:
            logger.error(f"Error getting collection stats: {e}")
            return {"error": str(e)} 