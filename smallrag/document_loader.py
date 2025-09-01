"""
Document loading and processing utilities for SmallRAG
"""

import os
from typing import List, Optional
from pathlib import Path
import re

from .utils import setup_logging

logger = setup_logging(__name__)


class DocumentLoader:
    """
    Handles loading and processing of various document formats
    """
    
    def __init__(self):
        """Initialize document loader"""
        self.supported_extensions = {
            '.txt': self._load_text,
            '.md': self._load_text,
            '.csv': self._load_csv,
            '.json': self._load_json,
            '.xml': self._load_xml,
            '.html': self._load_html
        }
    
    def load_document(self, file_path: str) -> str:
        """
        Load a document from file
        
        Args:
            file_path: Path to the document file
            
        Returns:
            Document content as string
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        extension = file_path.suffix.lower()
        
        if extension not in self.supported_extensions:
            raise ValueError(f"Unsupported file extension: {extension}")
        
        try:
            content = self.supported_extensions[extension](file_path)
            logger.info(f"Successfully loaded document: {file_path}")
            return content
        except Exception as e:
            logger.error(f"Error loading document {file_path}: {e}")
            raise
    
    def load_documents(self, file_paths: List[str]) -> List[str]:
        """
        Load multiple documents
        
        Args:
            file_paths: List of file paths
            
        Returns:
            List of document contents
        """
        documents = []
        
        for file_path in file_paths:
            try:
                content = self.load_document(file_path)
                documents.append(content)
            except Exception as e:
                logger.warning(f"Skipping {file_path}: {e}")
                continue
        
        return documents
    
    def _load_text(self, file_path: Path) -> str:
        """Load text file"""
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    
    def _load_csv(self, file_path: Path) -> str:
        """Load CSV file and convert to text"""
        import pandas as pd
        
        df = pd.read_csv(file_path)
        
        # Convert DataFrame to text
        text_parts = []
        
        # Add column names
        text_parts.append("Columns: " + ", ".join(df.columns.tolist()))
        
        # Add data as text
        for idx, row in df.iterrows():
            row_text = " ".join([f"{col}: {val}" for col, val in row.items()])
            text_parts.append(f"Row {idx}: {row_text}")
        
        return "\n".join(text_parts)
    
    def _load_json(self, file_path: Path) -> str:
        """Load JSON file and convert to text"""
        import json
        
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        return self._json_to_text(data)
    
    def _json_to_text(self, data, indent: int = 0) -> str:
        """Convert JSON data to readable text"""
        if isinstance(data, dict):
            text_parts = []
            for key, value in data.items():
                if isinstance(value, (dict, list)):
                    text_parts.append(f"{key}: {self._json_to_text(value, indent + 1)}")
                else:
                    text_parts.append(f"{key}: {value}")
            return "\n".join(text_parts)
        elif isinstance(data, list):
            text_parts = []
            for i, item in enumerate(data):
                if isinstance(item, (dict, list)):
                    text_parts.append(f"Item {i}: {self._json_to_text(item, indent + 1)}")
                else:
                    text_parts.append(f"Item {i}: {item}")
            return "\n".join(text_parts)
        else:
            return str(data)
    
    def _load_xml(self, file_path: Path) -> str:
        """Load XML file and convert to text"""
        import xml.etree.ElementTree as ET
        
        tree = ET.parse(file_path)
        root = tree.getroot()
        
        return self._xml_to_text(root)
    
    def _xml_to_text(self, element, indent: int = 0) -> str:
        """Convert XML element to readable text"""
        text_parts = []
        
        # Add element tag
        text_parts.append(f"{'  ' * indent}{element.tag}")
        
        # Add attributes
        if element.attrib:
            attrs = " ".join([f"{k}='{v}'" for k, v in element.attrib.items()])
            text_parts.append(f"{'  ' * (indent + 1)}Attributes: {attrs}")
        
        # Add text content
        if element.text and element.text.strip():
            text_parts.append(f"{'  ' * (indent + 1)}Content: {element.text.strip()}")
        
        # Process children
        for child in element:
            text_parts.append(self._xml_to_text(child, indent + 1))
        
        return "\n".join(text_parts)
    
    def _load_html(self, file_path: Path) -> str:
        """Load HTML file and extract text"""
        from bs4 import BeautifulSoup
        
        with open(file_path, 'r', encoding='utf-8') as f:
            soup = BeautifulSoup(f.read(), 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()
        
        # Get text
        text = soup.get_text()
        
        # Clean up whitespace
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = ' '.join(chunk for chunk in chunks if chunk)
        
        return text
    
    def extract_text_from_directory(self, directory: str, recursive: bool = True) -> List[str]:
        """
        Extract text from all supported files in a directory
        
        Args:
            directory: Directory path
            recursive: Whether to search subdirectories
            
        Returns:
            List of document contents
        """
        directory = Path(directory)
        
        if not directory.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")
        
        documents = []
        
        if recursive:
            file_paths = directory.rglob("*")
        else:
            file_paths = directory.glob("*")
        
        for file_path in file_paths:
            if file_path.is_file() and file_path.suffix.lower() in self.supported_extensions:
                try:
                    content = self.load_document(str(file_path))
                    documents.append(content)
                except Exception as e:
                    logger.warning(f"Skipping {file_path}: {e}")
                    continue
        
        return documents
    
    def preprocess_text(self, text: str) -> str:
        """
        Basic text preprocessing
        
        Args:
            text: Raw text
            
        Returns:
            Preprocessed text
        """
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s\.\,\!\?\;\:\-\(\)]', '', text)
        
        # Normalize whitespace
        text = ' '.join(text.split())
        
        return text.strip()
    
    def split_documents(self, documents: List[str], max_length: int = 1000) -> List[str]:
        """
        Split documents into smaller chunks
        
        Args:
            documents: List of document texts
            max_length: Maximum length of each chunk
            
        Returns:
            List of document chunks
        """
        chunks = []
        
        for doc in documents:
            if len(doc) <= max_length:
                chunks.append(doc)
            else:
                # Split by sentences or paragraphs
                sentences = re.split(r'[.!?]+', doc)
                current_chunk = ""
                
                for sentence in sentences:
                    if len(current_chunk) + len(sentence) <= max_length:
                        current_chunk += sentence + ". "
                    else:
                        if current_chunk:
                            chunks.append(current_chunk.strip())
                        current_chunk = sentence + ". "
                
                if current_chunk:
                    chunks.append(current_chunk.strip())
        
        return chunks 