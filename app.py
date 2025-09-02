"""
Streamlit web interface for SmallRAG with LDA
"""

import streamlit as st 
import os
import tempfile
from pathlib import Path
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from smallrag import SmallRAG, LDAModel, DocumentLoader
from smallrag.utils import setup_environment, preprocess_text

# Setup page config
st.set_page_config(
    page_title="SmallRAG with LDA",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Setup environment
setup_environment()

def main():
    st.title("🔍 SmallRAG with LDA Topic Modeling")
    st.markdown("A lightweight Retrieval-Augmented Generation system with Latent Dirichlet Allocation")
    
    # Sidebar
    st.sidebar.header("Configuration")
    
    # Model selection
    model_type = st.sidebar.selectbox(
        "Select Model Type",
        ["RAG System", "LDA Topic Modeling", "Combined Analysis"]
    )
    
    if model_type == "RAG System":
        rag_interface()
    elif model_type == "LDA Topic Modeling":
        lda_interface()
    else:
        combined_interface()

def rag_interface():
    st.header("📚 RAG System")
    
    # Initialize RAG
    if 'rag' not in st.session_state:
        st.session_state.rag = SmallRAG()
    
    # File upload
    st.subheader("Upload Documents")
    uploaded_files = st.file_uploader(
        "Choose files",
        type=['txt', 'md', 'csv', 'json', 'xml', 'html'],
        accept_multiple_files=True
    )
    
    if uploaded_files:
        # Save uploaded files temporarily
        temp_files = []
        for uploaded_file in uploaded_files:
            with tempfile.NamedTemporaryFile(delete=False, suffix=uploaded_file.name) as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                temp_files.append(tmp_file.name)
        
        # Add documents to RAG
        if st.button("Process Documents"):
            with st.spinner("Processing documents..."):
                try:
                    st.session_state.rag.add_documents(temp_files)
                    st.success(f"Processed {len(uploaded_files)} documents!")
                except Exception as e:
                    st.error(f"Error processing documents: {e}")
                finally:
                    # Clean up temp files
                    for temp_file in temp_files:
                        os.unlink(temp_file)
    
    # Query interface
    st.subheader("Ask Questions")
    query = st.text_input("Enter your question:")
    
    if query and st.button("Search"):
        with st.spinner("Searching..."):
            try:
                result = st.session_state.rag.query(query)
                
                st.subheader("Answer")
                st.write(result["answer"])
                
                if result.get("sources"):
                    st.subheader("Sources")
                    for source in result["sources"]:
                        st.write(f"- {source}")
                        
            except Exception as e:
                st.error(f"Error during query: {e}")
    
    # Similarity search
    st.subheader("Similarity Search")
    search_query = st.text_input("Enter search query:")
    k_results = st.slider("Number of results", 1, 10, 5)
    
    if search_query and st.button("Find Similar"):
        with st.spinner("Searching..."):
            try:
                similar_docs = st.session_state.rag.similarity_search(search_query, k=k_results)
                
                for i, doc in enumerate(similar_docs):
                    st.write(f"**Result {i+1}:**")
                    st.write(doc.page_content[:200] + "...")
                    st.write(f"*Source: {doc.metadata.get('source', 'Unknown')}*")
                    st.divider()
                    
            except Exception as e:
                st.error(f"Error during similarity search: {e}")

def lda_interface():
    st.header("📊 LDA Topic Modeling")
    
    # Initialize LDA
    if 'lda' not in st.session_state:
        st.session_state.lda = LDAModel()
    
    # Parameters
    col1, col2, col3 = st.columns(3)
    with col1:
        n_topics = st.number_input("Number of Topics", 2, 20, 5)
    with col2:
        random_state = st.number_input("Random State", 0, 100, 42)
    with col3:
        use_gensim = st.checkbox("Use Gensim (Recommended)", True)
    
    # File upload for LDA
    st.subheader("Upload Documents for Topic Modeling")
    uploaded_files = st.file_uploader(
        "Choose files for LDA",
        type=['txt', 'md', 'csv', 'json', 'xml', 'html'],
        accept_multiple_files=True,
        key="lda_upload"
    )
    
    if uploaded_files:
        # Save and load documents
        temp_files = []
        documents = []
        
        for uploaded_file in uploaded_files:
            with tempfile.NamedTemporaryFile(delete=False, suffix=uploaded_file.name) as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                temp_files.append(tmp_file.name)
        
        # Load documents
        loader = DocumentLoader()
        for temp_file in temp_files:
            try:
                content = loader.load_document(temp_file)
                documents.append(content)
            except Exception as e:
                st.warning(f"Error loading {temp_file}: {e}")
            finally:
                os.unlink(temp_file)
        
        if documents:
            st.write(f"Loaded {len(documents)} documents")
            
            # Train LDA model
            if st.button("Train LDA Model"):
                with st.spinner("Training LDA model..."):
                    try:
                        st.session_state.lda = LDAModel(n_topics=n_topics, random_state=random_state)
                        st.session_state.lda.fit(documents, use_gensim=use_gensim)
                        st.success("LDA model trained successfully!")
                    except Exception as e:
                        st.error(f"Error training model: {e}")
    
    # Display topics
    if hasattr(st.session_state.lda, 'gensim_lda') and st.session_state.lda.gensim_lda:
        st.subheader("Discovered Topics")
        
        topics = st.session_state.lda.get_topics()
        
        for topic in topics:
            with st.expander(f"Topic {topic['topic_id']}"):
                # Create word cloud
                word_weights = dict(zip(topic['words'], topic['weights']))
                
                # Display words and weights
                for word, weight in zip(topic['words'], topic['weights']):
                    st.write(f"**{word}**: {weight:.4f}")
    
    # Document topic analysis
    st.subheader("Document Topic Analysis")
    doc_text = st.text_area("Enter document text for topic analysis:")
    
    if doc_text and hasattr(st.session_state.lda, 'gensim_lda') and st.session_state.lda.gensim_lda:
        if st.button("Analyze Topics"):
            with st.spinner("Analyzing..."):
                try:
                    topic_dist = st.session_state.lda.get_document_topics(doc_text)
                    
                    # Create visualization
                    topic_ids = [topic_id for topic_id, _ in topic_dist]
                    probabilities = [prob for _, prob in topic_dist]
                    
                    fig = go.Figure(data=[
                        go.Bar(x=[f"Topic {tid}" for tid in topic_ids], y=probabilities)
                    ])
                    fig.update_layout(
                        title="Document Topic Distribution",
                        xaxis_title="Topics",
                        yaxis_title="Probability"
                    )
                    st.plotly_chart(fig)
                    
                except Exception as e:
                    st.error(f"Error analyzing topics: {e}")

def combined_interface():
    st.header("🔗 Combined RAG + LDA Analysis")
    
    st.info("This interface combines RAG retrieval with LDA topic modeling for enhanced document analysis.")
    
    # File upload
    uploaded_files = st.file_uploader(
        "Choose files for combined analysis",
        type=['txt', 'md', 'csv', 'json', 'xml', 'html'],
        accept_multiple_files=True,
        key="combined_upload"
    )
    
    if uploaded_files:
        # Process documents
        if st.button("Process for Combined Analysis"):
            with st.spinner("Processing documents..."):
                try:
                    # Initialize models
                    rag = SmallRAG()
                    lda = LDAModel()
                    
                    # Save and process files
                    temp_files = []
                    documents = []
                    
                    for uploaded_file in uploaded_files:
                        with tempfile.NamedTemporaryFile(delete=False, suffix=uploaded_file.name) as tmp_file:
                            tmp_file.write(uploaded_file.getvalue())
                            temp_files.append(tmp_file.name)
                    
                    # Load documents
                    loader = DocumentLoader()
                    for temp_file in temp_files:
                        try:
                            content = loader.load_document(temp_file)
                            documents.append(content)
                        except Exception as e:
                            st.warning(f"Error loading {temp_file}: {e}")
                        finally:
                            os.unlink(temp_file)
                    
                    if documents:
                        # Train LDA
                        lda.fit(documents)
                        
                        # Add to RAG
                        rag.add_documents(temp_files)
                        
                        st.success("Documents processed for both RAG and LDA!")
                        
                        # Store in session state
                        st.session_state.combined_rag = rag
                        st.session_state.combined_lda = lda
                        
                except Exception as e:
                    st.error(f"Error in combined processing: {e}")
    
    # Combined query
    if 'combined_rag' in st.session_state and 'combined_lda' in st.session_state:
        st.subheader("Combined Query")
        query = st.text_input("Enter your question:")
        
        if query and st.button("Analyze"):
            with st.spinner("Analyzing..."):
                try:
                    # RAG response
                    rag_result = st.session_state.combined_rag.query(query)
                    
                    # LDA topic analysis of query
                    topic_dist = st.session_state.combined_lda.get_document_topics(query)
                    
                    # Display results
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("RAG Answer")
                        st.write(rag_result["answer"])
                        
                        if rag_result.get("sources"):
                            st.write("**Sources:**")
                            for source in rag_result["sources"]:
                                st.write(f"- {source}")
                    
                    with col2:
                        st.subheader("Query Topic Analysis")
                        if topic_dist:
                            topic_ids = [topic_id for topic_id, _ in topic_dist]
                            probabilities = [prob for _, prob in topic_dist]
                            
                            fig = go.Figure(data=[
                                go.Bar(x=[f"Topic {tid}" for tid in topic_ids], y=probabilities)
                            ])
                            fig.update_layout(
                                title="Query Topic Distribution",
                                xaxis_title="Topics",
                                yaxis_title="Probability"
                            )
                            st.plotly_chart(fig)
                        else:
                            st.write("No topic distribution available")
                            
                except Exception as e:
                    st.error(f"Error in combined analysis: {e}")

if __name__ == "__main__":
    main() 