# Simple Embedding Project

![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![LangChain](https://img.shields.io/badge/LangChain-Latest-green.svg)

A simple yet powerful document embedding and retrieval system built with LangChain, ChromaDB, and Sentence Transformers. This project demonstrates how to process PDF documents, generate embeddings, store them in a vector database, and perform semantic search queries.

## 🚀 Features

- **PDF Processing**: Load and parse PDF documents using PyPDF
- **Text Chunking**: Split documents into manageable chunks for better embedding quality
- **Embedding Generation**: Create semantic embeddings using the `all-MiniLM-L6-v2` model from Sentence Transformers
- **Vector Storage**: Store embeddings in ChromaDB for efficient retrieval
- **Semantic Search**: Query the vector database to retrieve relevant document chunks
- **LangChain Integration**: Leverages LangChain for seamless workflow orchestration

## 📋 Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- Git (for cloning the repository)

## 🛠️ Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/Simple-Embedding-Project.git
   cd Simple-Embedding-Project
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   ```

3. **Activate the virtual environment**
   
   On Windows:
   ```bash
   venv\Scripts\activate
   ```
   
   On macOS/Linux:
   ```bash
   source venv/bin/activate
   ```

4. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## 📦 Dependencies

```
langchain
langchain-core
langchain-community
langchain-chroma
langchain-huggingface
pypdf
chromadb
sentence-transformers
python-dotenv
typesense
```

## 🎯 Usage

### Basic Usage

1. **Place your PDF file** in the project directory

2. **Run the main script**
   ```bash
   python main.py
   ```


## 🔍 How It Works

1. **Document Loading**: The system loads PDF documents using PyPDFLoader
2. **Text Splitting**: Documents are split into smaller chunks using CharacterTextSplitter for optimal embedding generation
3. **Embedding Generation**: Each chunk is converted into a 384-dimensional vector using the `all-MiniLM-L6-v2` model
4. **Vector Storage**: Embeddings are stored in ChromaDB, a vector database optimized for similarity search
5. **Retrieval**: When a query is made, it's converted to an embedding and compared against stored vectors to find the most relevant chunks

```

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [LangChain](https://github.com/langchain-ai/langchain) - Framework for developing applications with LLMs
- [ChromaDB](https://www.trychroma.com/) - AI-native open-source embedding database
- [Sentence Transformers](https://www.sbert.net/) - Framework for state-of-the-art sentence embeddings
- [Hugging Face](https://huggingface.co/) - ML model repository


