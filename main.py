
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
import numpy as np
from typing import List, Dict, Any, Tuple
from sklearn.metrics.pairwise import cosine_similarity

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

loader = PyPDFLoader('iot-paper.pdf')
docs = loader.load()

splitter = CharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=20,
    separator=''
)
result = splitter.split_documents(docs)
#print(result[10].page_content)

"""Load the SentenceTransformer model"""

# Use HuggingFaceEmbeddings (easiest solution)
embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

#print(f"Model loaded successfully. Embedding dimension: {embeddings.client.get_sentence_embedding_dimension()}")

# Create vector store with the new import
vector_store = Chroma(
    embedding_function=embeddings,
    persist_directory='my_chroma_db',
    collection_name='sample'
)

# Add documents
vector_store.add_documents(result)

output = vector_store.similarity_search(
    query='introduction?',
    k=2
)
print(output[0].page_content)
print(output[1].page_content)


