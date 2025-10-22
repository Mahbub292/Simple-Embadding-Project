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

### Example Code

```python
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# Load PDF
loader = PyPDFLoader("your_document.pdf")
docs = loader.load()

# Split into chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
chunks = splitter.split_documents(docs)

# Create embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

# Store in ChromaDB
vector_store = Chroma(
    embedding_function=embeddings,
    persist_directory='my_chroma_db',
    collection_name='sample'
)
vector_store.add_documents(chunks)

# Query the database
query = "What is machine learning?"
results = vector_store.similarity_search(query, k=3)

for doc in results:
    print(doc.page_content)
```

## 📁 Project Structure

```
Simple-Embedding-Project/
│
├── main.py                 # Main application script
├── requirements.txt        # Python dependencies
├── .gitignore             # Git ignore file
├── LICENSE                # MIT License
├── README.md              # Project documentation
│
├── venv/                  # Virtual environment (not tracked)
├── my_chroma_db/          # ChromaDB storage (not tracked)
│
└── data/                  # Directory for PDF files (optional)
```

## 🔍 How It Works

1. **Document Loading**: The system loads PDF documents using PyPDFLoader
2. **Text Splitting**: Documents are split into smaller chunks using RecursiveCharacterTextSplitter for optimal embedding generation
3. **Embedding Generation**: Each chunk is converted into a 384-dimensional vector using the `all-MiniLM-L6-v2` model
4. **Vector Storage**: Embeddings are stored in ChromaDB, a vector database optimized for similarity search
5. **Retrieval**: When a query is made, it's converted to an embedding and compared against stored vectors to find the most relevant chunks

## 🧪 Example Queries

```python
# Search for specific information
results = vector_store.similarity_search("What is deep learning?", k=3)

# Search with score threshold
results = vector_store.similarity_search_with_score("neural networks", k=5)

# Get relevant documents
retriever = vector_store.as_retriever(search_kwargs={"k": 3})
docs = retriever.get_relevant_documents("transformer architecture")
```

## ⚙️ Configuration

You can customize the following parameters in `main.py`:

- **Chunk Size**: Adjust `chunk_size` in RecursiveCharacterTextSplitter (default: 1000)
- **Chunk Overlap**: Modify `chunk_overlap` for context continuity (default: 200)
- **Embedding Model**: Change to other Sentence Transformer models
- **Collection Name**: Set a custom name for your ChromaDB collection
- **Number of Results**: Adjust `k` parameter in similarity search

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the project
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [LangChain](https://github.com/langchain-ai/langchain) - Framework for developing applications with LLMs
- [ChromaDB](https://www.trychroma.com/) - AI-native open-source embedding database
- [Sentence Transformers](https://www.sbert.net/) - Framework for state-of-the-art sentence embeddings
- [Hugging Face](https://huggingface.co/) - ML model repository

## 📧 Contact

Your Name - [@yourtwitter](https://twitter.com/yourtwitter) - your.email@example.com

Project Link: [https://github.com/yourusername/Simple-Embedding-Project](https://github.com/yourusername/Simple-Embedding-Project)

## 🐛 Troubleshooting

### Common Issues

**Issue**: `AttributeError: 'SentenceTransformer' object has no attribute 'embed_documents'`
- **Solution**: Use `HuggingFaceEmbeddings` from `langchain_huggingface` instead of raw SentenceTransformer

**Issue**: `git add .` takes too long
- **Solution**: Ensure `.gitignore` is properly excluding `venv/` and `my_chroma_db/` folders

**Issue**: ChromaDB persistence not working
- **Solution**: Check that `persist_directory` path exists and has write permissions

## 🔮 Future Enhancements

- [ ] Add support for multiple document formats (DOCX, TXT, etc.)
- [ ] Implement query result ranking and filtering
- [ ] Add a web interface using Streamlit or Gradio
- [ ] Support for multiple embedding models
- [ ] Add metadata filtering capabilities
- [ ] Implement document update and deletion features
