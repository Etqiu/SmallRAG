"""
LDA (Latent Dirichlet Allocation) topic modeling for SmallRAG
"""

import os
import pickle
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.metrics import silhouette_score
import gensim
from gensim import corpora, models
from gensim.models import LdaModel
from gensim.utils import simple_preprocess
import pyLDAvis
import pyLDAvis.gensim_models
from wordcloud import WordCloud

from .utils import setup_logging, preprocess_text

logger = setup_logging(__name__)


class LDAModel:
    """
    LDA topic modeling implementation for document analysis
    """
    
    def __init__(
        self,
        n_topics: int = 5,
        random_state: int = 42,
        max_iter: int = 10,
        learning_method: str = 'batch',
        model_save_path: str = "./models/lda_model.pkl"
    ):
        """
        Initialize LDA model
        
        Args:
            n_topics: Number of topics to discover
            random_state: Random seed for reproducibility
            max_iter: Maximum iterations for training
            learning_method: Learning method ('batch' or 'online')
            model_save_path: Path to save the trained model
        """
        self.n_topics = n_topics
        self.random_state = random_state
        self.max_iter = max_iter
        self.learning_method = learning_method
        self.model_save_path = model_save_path
        
        self.vectorizer = None
        self.lda_model = None
        self.gensim_lda = None
        self.dictionary = None
        self.corpus = None
        self.documents = []
        self.processed_docs = []
        
        os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
        
    def preprocess_documents(self, documents: List[str]) -> List[str]:
        """
        Preprocess documents for LDA analysis
        
        Args:
            documents: List of raw document texts
            
        Returns:
            List of preprocessed document texts
        """
        processed_docs = []
        
        for doc in documents:
            processed = preprocess_text(doc)
            processed_docs.append(processed)
            
        self.processed_docs = processed_docs
        logger.info(f"Preprocessed {len(processed_docs)} documents")
        return processed_docs
    
    def fit(self, documents: List[str], use_gensim: bool = True) -> None:
        """
        Fit LDA model to documents
        
        Args:
            documents: List of document texts
            use_gensim: Whether to use Gensim implementation (recommended)
        """
        self.documents = documents
        
        if use_gensim:
            self._fit_gensim(documents)
        else:
            self._fit_sklearn(documents)
            
        logger.info(f"Fitted LDA model with {self.n_topics} topics")
    
    def _fit_gensim(self, documents: List[str]) -> None:
        """Fit Gensim LDA model"""
        processed_docs = self.preprocess_documents(documents)
        
        self.dictionary = corpora.Dictionary(processed_docs)
        
        self.dictionary.filter_extremes(no_below=2, no_above=0.5)
        
        self.corpus = [self.dictionary.doc2bow(doc) for doc in processed_docs]
        
        self.gensim_lda = LdaModel(
            corpus=self.corpus,
            id2word=self.dictionary,
            num_topics=self.n_topics,
            random_state=self.random_state,
            update_every=1,
            chunksize=100,
            passes=10,
            alpha='auto',
            per_word_topics=True
        )
        
        logger.info("Trained Gensim LDA model")
    
    def _fit_sklearn(self, documents: List[str]) -> None:
        """Fit scikit-learn LDA model"""
        processed_docs = self.preprocess_documents(documents)
        
        self.vectorizer = CountVectorizer(
            max_df=0.95,
            min_df=2,
            stop_words='english',
            max_features=1000
        )
        
        doc_term_matrix = self.vectorizer.fit_transform(processed_docs)
        
        self.lda_model = LatentDirichletAllocation(
            n_components=self.n_topics,
            random_state=self.random_state,
            max_iter=self.max_iter,
            learning_method=self.learning_method
        )
        
        self.lda_model.fit(doc_term_matrix)
        
        logger.info("Trained scikit-learn LDA model")
    
    def get_topics(self, top_words: int = 10) -> List[Dict[str, Any]]:
        """
        Get discovered topics
        
        Args:
            top_words: Number of top words per topic
            
        Returns:
            List of topic dictionaries
        """
        if self.gensim_lda:
            return self._get_gensim_topics(top_words)
        elif self.lda_model:
            return self._get_sklearn_topics(top_words)
        else:
            logger.error("No trained model available")
            return []
    
    def _get_gensim_topics(self, top_words: int) -> List[Dict[str, Any]]:
        """Get topics from Gensim model"""
        topics = []
        
        for topic_id in range(self.n_topics):
            topic_words = self.gensim_lda.show_topic(topic_id, top_words)
            topics.append({
                'topic_id': topic_id,
                'words': [word for word, _ in topic_words],
                'weights': [weight for _, weight in topic_words]
            })
        
        return topics
    
    def _get_sklearn_topics(self, top_words: int) -> List[Dict[str, Any]]:
        """Get topics from scikit-learn model"""
        topics = []
        feature_names = self.vectorizer.get_feature_names_out()
        
        for topic_idx, topic in enumerate(self.lda_model.components_):
            top_word_indices = topic.argsort()[-top_words:][::-1]
            top_words_list = [feature_names[i] for i in top_word_indices]
            top_weights = [topic[i] for i in top_word_indices]
            
            topics.append({
                'topic_id': topic_idx,
                'words': top_words_list,
                'weights': top_weights.tolist()
            })
        
        return topics
    
    def get_document_topics(self, document: str) -> List[Tuple[int, float]]:
        """
        Get topic distribution for a document
        
        Args:
            document: Document text
            
        Returns:
            List of (topic_id, probability) tuples
        """
        if self.gensim_lda:
            processed_doc = simple_preprocess(document)
            
            doc_bow = self.dictionary.doc2bow(processed_doc)
            
            topic_dist = self.gensim_lda.get_document_topics(doc_bow)
            return topic_dist
        else:
            logger.error("Gensim model required for document topic analysis")
            return []
    
    def visualize_topics(self, save_path: Optional[str] = None) -> None:
        """
        Create interactive topic visualization
        
        Args:
            save_path: Path to save the visualization HTML file
        """
        if not self.gensim_lda:
            logger.error("Gensim model required for visualization")
            return
        
        try:
            vis_data = pyLDAvis.gensim_models.prepare(
                self.gensim_lda,
                self.corpus,
                self.dictionary,
                sort_topics=False
            )
            
            if save_path:
                pyLDAvis.save_html(vis_data, save_path)
                logger.info(f"Saved topic visualization to {save_path}")
            else:
                return pyLDAvis.prepare(
                    self.gensim_lda,
                    self.corpus,
                    self.dictionary,
                    sort_topics=False
                )
                
        except Exception as e:
            logger.error(f"Error creating visualization: {e}")
    
    def create_wordclouds(self, save_dir: str = "./models/wordclouds") -> None:
        """
        Create word clouds for each topic
        
        Args:
            save_dir: Directory to save word cloud images
        """
        os.makedirs(save_dir, exist_ok=True)
        
        topics = self.get_topics()
        
        for topic in topics:
            wordcloud = WordCloud(
                width=800,
                height=400,
                background_color='white',
                colormap='viridis'
            ).generate_from_frequencies(
                dict(zip(topic['words'], topic['weights']))
            )
            
            plt.figure(figsize=(10, 5))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis('off')
            plt.title(f'Topic {topic["topic_id"]}')
            
            save_path = os.path.join(save_dir, f"topic_{topic['topic_id']}.png")
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            plt.close()
            
        logger.info(f"Created word clouds in {save_dir}")
    
    def plot_topic_distribution(self, save_path: Optional[str] = None) -> None:
        """
        Plot topic distribution across documents
        
        Args:
            save_path: Path to save the plot
        """
        if not self.gensim_lda:
            logger.error("Gensim model required for topic distribution analysis")
            return
        
        doc_topics = []
        for doc_bow in self.corpus:
            topic_dist = self.gensim_lda.get_document_topics(doc_bow)
            doc_topics.append([prob for _, prob in topic_dist])
        
        df = pd.DataFrame(doc_topics, columns=[f'Topic {i}' for i in range(self.n_topics)])
        
        plt.figure(figsize=(12, 6))
        df.mean().plot(kind='bar')
        plt.title('Average Topic Distribution Across Documents')
        plt.xlabel('Topics')
        plt.ylabel('Average Probability')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
        
        plt.close()
    
    def save_model(self) -> None:
        """Save the trained model"""
        try:
            model_data = {
                'n_topics': self.n_topics,
                'random_state': self.random_state,
                'gensim_lda': self.gensim_lda,
                'dictionary': self.dictionary,
                'corpus': self.corpus,
                'documents': self.documents,
                'processed_docs': self.processed_docs
            }
            
            with open(self.model_save_path, 'wb') as f:
                pickle.dump(model_data, f)
            
            logger.info(f"Saved LDA model to {self.model_save_path}")
            
        except Exception as e:
            logger.error(f"Error saving model: {e}")
    
    def load_model(self) -> bool:
        """Load a previously saved model"""
        try:
            if not os.path.exists(self.model_save_path):
                logger.warning(f"Model file not found: {self.model_save_path}")
                return False
            
            with open(self.model_save_path, 'rb') as f:
                model_data = pickle.load(f)
            
            self.n_topics = model_data['n_topics']
            self.random_state = model_data['random_state']
            self.gensim_lda = model_data['gensim_lda']
            self.dictionary = model_data['dictionary']
            self.corpus = model_data['corpus']
            self.documents = model_data['documents']
            self.processed_docs = model_data['processed_docs']
            
            logger.info(f"Loaded LDA model from {self.model_save_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False
    
    def evaluate_model(self) -> Dict[str, float]:
        """
        Evaluate the LDA model quality
        
        Returns:
            Dictionary with evaluation metrics
        """
        if not self.gensim_lda:
            logger.error("No trained model available for evaluation")
            return {}
        
        try:
            perplexity = self.gensim_lda.log_perplexity(self.corpus)
            
            coherence_model = gensim.models.CoherenceModel(
                model=self.gensim_lda,
                texts=self.processed_docs,
                dictionary=self.dictionary,
                coherence='c_v'
            )
            coherence = coherence_model.get_coherence()
            
            return {
                'perplexity': perplexity,
                'coherence': coherence,
                'n_topics': self.n_topics
            }
            
        except Exception as e:
            logger.error(f"Error evaluating model: {e}")
            return {} 