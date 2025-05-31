import os
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Load your API key from the .env file
load_dotenv()
openai_key = os.getenv("OPENAI_API_KEY")

if not openai_key:
    raise Exception("❌ OPENAI_API_KEY not found. Check your .env file!")

# Path to your PDF manuals
manual_files = ["fluent_hardware_manual.pdf", "fluentcontrol_manual.pdf"]

all_docs = []

# Read and extract text from each PDF
for file in manual_files:
    loader = PyPDFLoader(file)
    docs = loader.load()
    all_docs.extend(docs)

print(f"Loaded {len(all_docs)} pages from {len(manual_files)} PDFs.")

# Split into smaller chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
chunks = text_splitter.split_documents(all_docs)

print(f"Split into {len(chunks)} chunks.")

# Create embeddings and store them in Chroma
embedding = OpenAIEmbeddings(openai_api_key=openai_key)
db = Chroma.from_documents(chunks, embedding, persist_directory="fluent_db")

print("✅ Embeddings saved to local vector database (fluent_db)")


