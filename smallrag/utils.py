"""
Utility functions for SmallRAG
"""

import os
import logging
import re
from typing import List, Optional
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')


def setup_logging(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Setup logging configuration
    
    Args:
        name: Logger name
        level: Logging level
        
    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    
    if not logger.handlers:
        logger.setLevel(level)
        
        # Create console handler
        handler = logging.StreamHandler()
        handler.setLevel(level)
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        
        # Add handler to logger
        logger.addHandler(handler)
    
    return logger


def preprocess_text(text: str, remove_stopwords: bool = True, lemmatize: bool = True) -> str:
    """
    Preprocess text for NLP tasks
    
    Args:
        text: Raw text
        remove_stopwords: Whether to remove stopwords
        lemmatize: Whether to lemmatize words
        
    Returns:
        Preprocessed text
    """
    # Convert to lowercase
    text = text.lower()
    
    # Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Tokenize
    tokens = word_tokenize(text)
    
    # Remove stopwords
    if remove_stopwords:
        stop_words = set(stopwords.words('english'))
        tokens = [token for token in tokens if token not in stop_words]
    
    # Lemmatize
    if lemmatize:
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(token) for token in tokens]
    
    # Remove short tokens
    tokens = [token for token in tokens if len(token) > 2]
    
    return ' '.join(tokens)


def extract_keywords(text: str, top_k: int = 10) -> List[str]:
    """
    Extract keywords from text using TF-IDF
    
    Args:
        text: Input text
        top_k: Number of keywords to extract
        
    Returns:
        List of keywords
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.feature_extraction.text import CountVectorizer
    
    # Preprocess text
    processed_text = preprocess_text(text)
    
    # Create TF-IDF vectorizer
    vectorizer = TfidfVectorizer(
        max_features=top_k,
        stop_words='english',
        ngram_range=(1, 2)
    )
    
    # Fit and transform
    tfidf_matrix = vectorizer.fit_transform([processed_text])
    
    # Get feature names
    feature_names = vectorizer.get_feature_names_out()
    
    # Get TF-IDF scores
    tfidf_scores = tfidf_matrix.toarray()[0]
    
    # Sort by score
    keyword_scores = list(zip(feature_names, tfidf_scores))
    keyword_scores.sort(key=lambda x: x[1], reverse=True)
    
    return [keyword for keyword, score in keyword_scores]


def calculate_text_similarity(text1: str, text2: str) -> float:
    """
    Calculate similarity between two texts using cosine similarity
    
    Args:
        text1: First text
        text2: Second text
        
    Returns:
        Similarity score (0-1)
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    
    # Create TF-IDF vectorizer
    vectorizer = TfidfVectorizer(stop_words='english')
    
    # Fit and transform
    tfidf_matrix = vectorizer.fit_transform([text1, text2])
    
    # Calculate cosine similarity
    similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
    
    return similarity


def create_summary(text: str, max_length: int = 150) -> str:
    """
    Create a simple extractive summary
    
    Args:
        text: Input text
        max_length: Maximum length of summary
        
    Returns:
        Summary text
    """
    # Split into sentences
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    if not sentences:
        return ""
    
    # Simple approach: take first few sentences
    summary = ""
    for sentence in sentences:
        if len(summary) + len(sentence) <= max_length:
            summary += sentence + ". "
        else:
            break
    
    return summary.strip()


def validate_file_path(file_path: str) -> bool:
    """
    Validate if a file path exists and is readable
    
    Args:
        file_path: Path to file
        
    Returns:
        True if file is valid, False otherwise
    """
    try:
        return os.path.isfile(file_path) and os.access(file_path, os.R_OK)
    except Exception:
        return False


def get_file_extension(file_path: str) -> str:
    """
    Get file extension from path
    
    Args:
        file_path: Path to file
        
    Returns:
        File extension (with dot)
    """
    return os.path.splitext(file_path)[1].lower()


def sanitize_filename(filename: str) -> str:
    """
    Sanitize filename for safe file operations
    
    Args:
        filename: Original filename
        
    Returns:
        Sanitized filename
    """
    # Remove or replace invalid characters
    invalid_chars = '<>:"/\\|?*'
    for char in invalid_chars:
        filename = filename.replace(char, '_')
    
    # Remove leading/trailing spaces and dots
    filename = filename.strip('. ')
    
    return filename


def format_file_size(size_bytes: int) -> str:
    """
    Format file size in human-readable format
    
    Args:
        size_bytes: Size in bytes
        
    Returns:
        Formatted size string
    """
    if size_bytes == 0:
        return "0B"
    
    size_names = ["B", "KB", "MB", "GB", "TB"]
    i = 0
    while size_bytes >= 1024 and i < len(size_names) - 1:
        size_bytes /= 1024.0
        i += 1
    
    return f"{size_bytes:.1f}{size_names[i]}"


def create_directory_if_not_exists(directory: str) -> None:
    """
    Create directory if it doesn't exist
    
    Args:
        directory: Directory path
    """
    if not os.path.exists(directory):
        os.makedirs(directory)


def get_environment_variable(key: str, default: Optional[str] = None) -> Optional[str]:
    """
    Get environment variable with fallback
    
    Args:
        key: Environment variable name
        default: Default value if not found
        
    Returns:
        Environment variable value or default
    """
    return os.getenv(key, default)


def setup_environment() -> None:
    """
    Setup environment for SmallRAG
    """
    # Create necessary directories
    directories = ['./data', './models', './chroma_db']
    for directory in directories:
        create_directory_if_not_exists(directory)
    
    # Check for required environment variables
    openai_key = get_environment_variable('OPENAI_API_KEY')
    if not openai_key:
        print("Warning: OPENAI_API_KEY not found in environment variables")
        print("Please set it to use the LLM features")
    
    print("SmallRAG environment setup complete!") 