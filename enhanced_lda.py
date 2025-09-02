"""
Enhanced LDA with Cluster Analysis and Visualization
"""

import os
import sys
import pickle
from typing import List, Dict, Any
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import re
import gensim
from gensim import corpora, models
from gensim.models import LdaModel
from gensim.utils import simple_preprocess
import nltk
from nltk.corpus import stopwords
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# Download required NLTK data
try:
    nltk.download('stopwords', quiet=True)
except:
    pass

def load_documents(directory: str) -> List[str]:
    """Load documents from directory and subdirectories"""
    documents = []
    
    if not os.path.exists(directory):
        print(f"Directory not found: {directory}")
        return documents
    
    # Search recursively in directory and subdirectories
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(('.txt', '.md')):  # Only load text files
                try:
                    file_path = os.path.join(root, file)
                    content = ""
                    
                    # Handle text files
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    if content.strip():  # Only add if content is not empty
                        documents.append(content)
                        # Show relative path from data directory
                        rel_path = os.path.relpath(file_path, directory)
                        print(f"Loaded: {rel_path}")
                    
                except Exception as e:
                    print(f"Error loading {file}: {e}")
    
    return documents

def load_documents_with_names(directory: str) -> (List[str], List[str]):
    """Load documents and their original filenames from directory and subdirectories"""
    documents = []
    doc_names = []
    
    if not os.path.exists(directory):
        print(f"Directory not found: {directory}")
        return documents, doc_names
    
    # Search recursively in directory and subdirectories
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(('.txt', '.md')):  # Only load text files
                try:
                    file_path = os.path.join(root, file)
                    content = ""
                    
                    # Handle text files
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    if content.strip():  # Only add if content is not empty
                        documents.append(content)
                        doc_names.append(file)  # Store the original filename
                        # Show relative path from data directory
                        rel_path = os.path.relpath(file_path, directory)
                        print(f"Loaded: {rel_path}")
                    
                except Exception as e:
                    print(f"Error loading {file}: {e}")
    
    return documents, doc_names

def preprocess_text(text: str) -> List[str]:
    """Simple text preprocessing"""
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    words = text.split()
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words and len(word) > 2]
    
    return words

class EnhancedLDA:
    """Enhanced LDA with clustering and visualization"""
    
    def __init__(self, n_topics: int = 10, random_state: int = 42):
        self.n_topics = n_topics
        self.random_state = random_state
        self.lda_model = None
        self.dictionary = None
        self.corpus = None
        self.document_names = []
        self.topic_distributions = None
        
    def fit(self, documents: List[str], document_names: List[str] = None):
        """Train LDA model"""
        print("Preprocessing documents...")
        
        # Preprocess documents
        processed_docs = []
        for doc in documents:
            words = preprocess_text(doc)
            if words:  # Only add if we have words
                processed_docs.append(words)
        
        print(f"Processed {len(processed_docs)} documents")
        
        # Create dictionary
        self.dictionary = corpora.Dictionary(processed_docs)
        
        # Filter extreme values
        self.dictionary.filter_extremes(no_below=2, no_above=0.5)
        
        if len(self.dictionary) == 0:
            print("Dictionary is empty after filtering!")
            return
        
        print(f"Dictionary size: {len(self.dictionary)}")
        
        # Create corpus (bow rep)
        self.corpus = [self.dictionary.doc2bow(doc) for doc in processed_docs]
        
        # Store names if possible
        if document_names:
            self.document_names = document_names[:len(processed_docs)]
        else:
            self.document_names = [f"Document_{i}" for i in range(len(processed_docs))]
        
        # Train LDA model
        print(f"Training LDA with {self.n_topics} topics.")
        self.lda_model = LdaModel(
            corpus=self.corpus,
            id2word=self.dictionary,
            num_topics=self.n_topics,
            random_state=self.random_state,
            passes=15,
            alpha='auto',
            per_word_topics=True
        )
        
        # Get topic distributions for all documents
        self.topic_distributions = []
        for doc_bow in self.corpus:
            topic_dist = self.lda_model.get_document_topics(doc_bow)
            # Create full distribution vector
            dist_vector = [0.0] * self.n_topics
            for topic_id, prob in topic_dist:
                dist_vector[topic_id] = prob
            self.topic_distributions.append(dist_vector)
        
        print("LDA training complete!")
    
    def get_topics(self, top_words: int = 15) -> List[Dict[str, Any]]:
        """Get discovered topics"""
        if not self.lda_model:
            return []
        
        topics = []
        for topic_id in range(self.n_topics):
            topic_words = self.lda_model.show_topic(topic_id, top_words)
            topics.append({
                'topic_id': topic_id,
                'words': [word for word, _ in topic_words],
                'weights': [weight for _, weight in topic_words]
            })
        
        return topics
    
    def cluster_documents(self, n_clusters: int = 10):
        """Cluster documents based on topic distributions"""
        if self.topic_distributions is None:
            print("No topic distributions available!")
            return None
        
        # Convert to numpy array
        topic_matrix = np.array(self.topic_distributions)
        
        # Apply K-means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=self.random_state)
        cluster_labels = kmeans.fit_predict(topic_matrix)
        
        # Create results DataFrame
        results = pd.DataFrame({
            'document': self.document_names,
            'cluster': cluster_labels
        })
        
        # Add topic distributions
        for i in range(self.n_topics):
            results[f'topic_{i}'] = topic_matrix[:, i]
        
        return results
    
    def visualize_topics(self, save_path: str = None):
        """Create comprehensive topic visualization"""
        if not self.lda_model:
            print("No trained model available!")
            return
        
        topics = self.get_topics()
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Topic Word Clouds', 'Topic Distribution', 'Document Clusters', 'Topic Similarity'),
            specs=[[{"type": "bar"}, {"type": "bar"}],
                   [{"type": "scatter"}, {"type": "heatmap"}]]
        )
        
        # 1. Topic word clouds (bar chart representation)
        for topic in topics[:5]:  # Show first 5 topics
            fig.add_trace(
                go.Bar(
                    x=topic['words'][:10],
                    y=topic['weights'][:10],
                    name=f"Topic {topic['topic_id']}",
                    showlegend=True
                ),
                row=1, col=1
            )
        
        # 2. Topic distribution across documents
        if self.topic_distributions:
            topic_means = np.mean(self.topic_distributions, axis=0)
            fig.add_trace(
                go.Bar(
                    x=[f"Topic {i}" for i in range(self.n_topics)],
                    y=topic_means,
                    name="Avg Distribution",
                    showlegend=False
                ),
                row=1, col=2
            )
        
        # 3. Document clusters (if available)
        if self.topic_distributions:

            pca = PCA(n_components=2) # PCA for dimensionality reduction
            topic_matrix = np.array(self.topic_distributions)
            reduced_data = pca.fit_transform(topic_matrix)
            
            # Apply clustering
            kmeans = KMeans(n_clusters=min(5, len(self.document_names)), random_state=self.random_state)
            clusters = kmeans.fit_predict(topic_matrix)
            
            fig.add_trace(
                go.Scatter(
                    x=reduced_data[:, 0],
                    y=reduced_data[:, 1],
                    mode='markers',
                    marker=dict(color=clusters, colorscale='viridis'),
                    text=self.document_names,
                    name="Documents",
                    showlegend=False
                ),
                row=2, col=1
            )
        
        # 4. Topic similarity heatmap
        if self.topic_distributions:
            topic_matrix = np.array(self.topic_distributions)
            similarity_matrix = np.corrcoef(topic_matrix.T)
            
            fig.add_trace(
                go.Heatmap(
                    z=similarity_matrix,
                    x=[f"Topic {i}" for i in range(self.n_topics)],
                    y=[f"Topic {i}" for i in range(self.n_topics)],
                    colorscale='RdBu',
                    showlegend=False
                ),
                row=2, col=2
            )
        
        # Update layout
        fig.update_layout(
            title="LDA Topic Analysis Dashboard",
            height=800,
            showlegend=True
        )
        
        if save_path:
            fig.write_html(save_path)
            print(f"Visualization saved to {save_path}")
        else:
            fig.show()
    
    def create_cluster_analysis(self, n_clusters: int = 5, save_path: str = None):
        """Create detailed cluster analysis"""
        if self.topic_distributions is None:
            print("No topic distributions available!")
            return
        
        cluster_results = self.cluster_documents(n_clusters)
        
        # Create visualization (4 subplots with plotly)
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Document Clusters', 'Cluster Sizes', 'Topic Distribution by Cluster', 'Cluster Characteristics'),
            specs=[[{"type": "scatter"}, {"type": "bar"}],
                   [{"type": "heatmap"}, {"type": "bar"}]]
        )
        
        # 1. Document clusters (PCA)
        pca = PCA(n_components=2)
        topic_matrix = np.array(self.topic_distributions)
        reduced_data = pca.fit_transform(topic_matrix)
        
        fig.add_trace(
            go.Scatter(
                x=reduced_data[:, 0],
                y=reduced_data[:, 1],
                mode='markers',
                marker=dict(
                    color=cluster_results['cluster'],
                    colorscale='viridis',
                    size=10
                ),
                text=cluster_results['document'],
                name="Documents",
                showlegend=False
            ),
            row=1, col=1
        )
        
        # 2. Cluster sizes
        cluster_sizes = cluster_results['cluster'].value_counts().sort_index()
        fig.add_trace(
            go.Bar(
                x=[f"Cluster {i}" for i in cluster_sizes.index],
                y=cluster_sizes.values,
                name="Cluster Sizes",
                showlegend=False
            ),
            row=1, col=2
        )
        
        # 3. Topic distribution by cluster
        topic_cols = [f'topic_{i}' for i in range(self.n_topics)]
        cluster_topic_means = cluster_results.groupby('cluster')[topic_cols].mean()
        
        fig.add_trace(
            go.Heatmap(
                z=cluster_topic_means.values,
                x=[f"Topic {i}" for i in range(self.n_topics)],
                y=[f"Cluster {i}" for i in cluster_topic_means.index],
                colorscale='Blues',
                showlegend=False
            ),
            row=2, col=1
        )
        
        # 4. Cluster characteristics (most prominent topics)
        cluster_topics = []
        for cluster_id in range(n_clusters):
            cluster_docs = cluster_results[cluster_results['cluster'] == cluster_id]
            if len(cluster_docs) > 0:
                topic_means = cluster_docs[topic_cols].mean()
                top_topic = topic_means.idxmax()
                cluster_topics.append((cluster_id, top_topic, topic_means[top_topic]))
        
        if cluster_topics:
            fig.add_trace(
                go.Bar(
                    x=[f"Cluster {c}" for c, _, _ in cluster_topics],
                    y=[v for _, _, v in cluster_topics],
                    text=[f"Topic {t.split('_')[1]}" for _, t, _ in cluster_topics],
                    name="Top Topics",
                    showlegend=False
                ),
                row=2, col=2
            )
        
        # Update layout
        fig.update_layout(
            title=f"Cluster Analysis (K={n_clusters})",
            height=800,
            showlegend=True
        )
        
        if save_path:
            fig.write_html(save_path)
            print(f"Cluster analysis saved to {save_path}")
        else:
            fig.show()
        
        return cluster_results
    
    def save_model(self, path: str = "./models/enhanced_lda_model.pkl"):
        """Save the trained model with topic keywords and document names"""
        os.makedirs(os.path.dirname(path), exist_ok=True)

        # Get topic keywords
        topic_keywords = []
        if self.lda_model:
            for topic_id in range(self.n_topics):
                topic_words = self.lda_model.show_topic(topic_id, topn=15)
                topic_keywords.append([word for word, _ in topic_words])

        model_data = {
            'n_topics': self.n_topics,
            'random_state': self.random_state,
            'lda_model': self.lda_model,
            'dictionary': self.dictionary,
            'corpus': self.corpus,
            'document_names': self.document_names,
            'topic_distributions': self.topic_distributions,
            'topic_keywords': topic_keywords  
        }

        with open(path, 'wb') as f:
            pickle.dump(model_data, f)

        print(f"Model saved to {path}")
    
    def save_cluster_results(self, cluster_results, path: str = "./models/visualizations/cluster_results.pkl"):
        """Save cluster results DataFrame as a pickle file"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(cluster_results, f)
        print(f"Cluster results saved to {path}")

def main():
    print("Enhanced LDA (different paramaters) with Cluster Analysis")
    print("=" * 50)
    
    # Load documents
    documents, document_names = load_documents_with_names("./data") 
    print(f"Loaded {len(documents)} documents")
    
    # Get parameters
    n_topics = int(input(f"\nNumber of topics (default: 10): ") or "10")
    n_clusters = int(input(f"Number of clusters (default: 10): ") or "10")
    
    lda = EnhancedLDA(n_topics=n_topics)
    lda.fit(documents, document_names=document_names)
    
    # Display topics
    print(f"\n Discovered Topics:")
    print("=" * 50)
    
    topics = lda.get_topics()
    for topic in topics:
        print(f"\n Topic {topic['topic_id']}:")
        print("-" * 30)
        for word, weight in zip(topic['words'][:10], topic['weights'][:10]):
            print(f"  • {word}: {weight:.4f}")
    
    os.makedirs("./models/visualizations", exist_ok=True)
    
    lda.visualize_topics("./models/visualizations/topic_analysis.html")
    
    cluster_results = lda.create_cluster_analysis(
        n_clusters=n_clusters, 
        save_path="./models/visualizations/cluster_analysis.html"
    )
    
    lda.save_model() # save the model with topic keywords and document names/filenames
    
    if cluster_results is not None:
        lda.save_cluster_results(cluster_results) 
        print(f"\nCluster Analysis Summary:")
        print("=" * 50)
        print(f"Number of clusters: {n_clusters}")
        print(f"Number of documents: {len(cluster_results)}")
        
        for cluster_id in range(n_clusters):
            cluster_docs = cluster_results[cluster_results['cluster'] == cluster_id]
            print(f"\n- Cluster {cluster_id}: {len(cluster_docs)} documents")
            if len(cluster_docs) > 0:
                # Show sample documents
                sample_docs = cluster_docs['document'].head(3).tolist()
                for doc in sample_docs:
                    print(f"   - {doc}")
    
    print(f"\n Enhanced LDA analysis complete!!!!!!")
    print(f" Visualizations saved to: ./models/visualizations/")

if __name__ == "__main__":
    main()