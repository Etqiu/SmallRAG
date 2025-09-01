"""
SmallRAG - A lightweight RAG system with LDA topic modeling
"""

__version__ = "0.1.0"
__author__ = "SmallRAG Team"

from .rag import SmallRAG
from .lda_model import LDAModel
from .document_loader import DocumentLoader

__all__ = ["SmallRAG", "LDAModel", "DocumentLoader"] 