import os
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import PGVector

# 1. Load and chunk the PDFs
print("📂 Loading and chunking PDFs...")
loader = PyPDFDirectoryLoader("./data")
docs = loader.load()

if not docs:
    print("❌ No PDFs found in the './data' folder. Please add your PDFs and try again.")
    exit()

# We split the text into smaller chunks so the AI can easily search through them
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.split_documents(docs)
print(f"✅ Created {len(chunks)} chunks of text from your PDFs.")

# 2. Initialize free HuggingFace embeddings
print("🧠 Loading embedding model (this might take a minute the first time)...")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# 3. Store in PostgreSQL
# We use localhost and your Mac user since we created the DB via Homebrew
CONNECTION_STRING = "postgresql+psycopg2://localhost/mamaspace_db"
COLLECTION_NAME = "mamaspace_docs"

print("💾 Storing embeddings in PostgreSQL...")
db = PGVector.from_documents(
    documents=chunks,
    embedding=embeddings,
    connection_string=CONNECTION_STRING,
    collection_name=COLLECTION_NAME,
)

print("🎉 Ingestion complete! Your clinical knowledge base is ready in PostgreSQL.")
