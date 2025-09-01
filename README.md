# SmallRAG with LDA Support

A lightweight Retrieval-Augmented Generation (RAG) system with Latent Dirichlet Allocation (LDA) topic modeling capabilities.

## Features

- **RAG Pipeline**: Document ingestion, embedding, and retrieval
- **LDA Topic Modeling**: Automatic topic discovery and visualization
- **Vector Database**: ChromaDB for efficient similarity search
- **Web Interface**: Streamlit-based demo application
- **Topic Visualization**: Interactive topic exploration

## Quick Start - LDA Only

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Download NLTK Data
```bash
python download_nltk.py
```

### 3. Run LDA Model
```bash
# Run with default 5 topics
python run_lda.py

# Run with custom number of topics
python run_lda.py 3
```

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd SmallRAG
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Download required NLP models:
```bash
python download_nltk.py
```

5. Set up environment variables (optional):
```bash
cp env_example.txt .env
# Edit .env with your OpenAI API key for RAG features
```

## Usage

### LDA Topic Modeling (Simplified)

```bash
# Run LDA on documents in data/ directory
python run_lda.py

# Specify number of topics
python run_lda.py 3
```

### Web Interface (Full RAG + LDA)

```bash
streamlit run app.py
```

## Project Structure

```
SmallRAG/
├── smallrag/              # Full RAG system (optional)
│   ├── __init__.py
│   ├── rag.py
│   ├── lda_model.py
│   ├── document_loader.py
│   └── utils.py
├── data/                  # Your documents go here
│   ├── machine_learning_overview.txt
│   └── nlp_overview.txt
├── models/               # Saved models
├── run_lda.py           # Simple LDA runner
├── download_nltk.py     # NLTK data downloader
├── app.py              # Streamlit web interface
├── requirements.txt    # Dependencies
└── README.md          # This file
```

## Where to Put Your Documents

Place your documents in the `data/` directory:

```
SmallRAG/
├── data/
│   ├── your_document1.txt
│   ├── your_document2.md
│   ├── your_document3.docx
│   └── your_document4.txt
```

**Supported formats**: `.txt`, `.md`, `.docx`

### Converting DOCX Files

If you have DOCX files (from Google Docs or Word):

```bash
# Convert DOCX files to text
python docx_converter.py

# Or install dependencies first
pip install -r requirements.txt
```

## Configuration

Create a `.env` file with:
```
OPENAI_API_KEY=your_openai_api_key_here
CHROMA_PERSIST_DIRECTORY=./chroma_db
```

## License

MIT License 