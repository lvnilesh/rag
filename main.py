# =============================
# RAG Pipeline with Local Ollama
# =============================

# Import required libraries
import os
import warnings
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain import hub
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_ollama import OllamaLLM, OllamaEmbeddings

# Suppress user warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)

# Load environment variables from .env file
load_dotenv()

# =============================
# 1. Load the Language Model (LLM) and Embeddings
# =============================

# Get model and base_url from environment variables
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.3")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# Initialize the Ollama LLM (make sure Ollama is running locally)
llm = OllamaLLM(model=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL)

# Initialize the Ollama Embeddings model
embed_model = OllamaEmbeddings(
    model=OLLAMA_MODEL,
    base_url=OLLAMA_BASE_URL
)

# =============================
# 2. Load and Prepare PDF Data
# =============================

# --- PDF ingestion version ---
# Directory containing PDF files
PDF_DIR = "/mnt/backup"

# List all PDF files in the directory
pdf_files = [os.path.join(PDF_DIR, f) for f in os.listdir(PDF_DIR) if f.lower().endswith('.pdf')]

# Load and combine all documents from PDFs
all_docs = []
for pdf_file in pdf_files:
    loader = PyPDFLoader(pdf_file)
    docs = loader.load()
    all_docs.extend(docs)

# =============================
# 3. Split PDF Documents into Chunks
# =============================

# Use a text splitter to break the documents into manageable chunks for embedding
text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=128)
chunks = text_splitter.split_documents(all_docs)

# =============================
# 4. Create a Vector Store (Chroma)
# =============================

# Embed the text chunks and store them in a local Chroma vector database
vector_store = Chroma.from_documents(chunks, embed_model)

# =============================
# --- Previous version: Example text version (commented out) ---
# =============================
# # 2. Prepare the Text Data
# # =============================
#
# # Example text to be used for retrieval-augmented generation (RAG)
# text = """
# In the lush canopy of a tropical rainforest, two mischievous monkeys, Coco and Mango, swung from branch to branch, their playful antics echoing through the trees. They were inseparable companions, sharing everything from juicy fruits to secret hideouts high above the forest floor. One day, while exploring a new part of the forest, Coco stumbled upon a beautiful orchid hidden among the foliage. Entranced by its delicate petals, Coco plucked it and presented it to Mango with a wide grin. Overwhelmed by Coco's gesture of friendship, Mango hugged Coco tightly, cherishing the bond they shared. From that day on, Coco and Mango ventured through the forest together, their friendship growing stronger with each passing adventure. As they watched the sun dip below the horizon, casting a golden glow over the treetops, they knew that no matter what challenges lay ahead, they would always have each other, and their hearts brimmed with joy.
# """
#
# # =============================
# # 3. Split Text into Chunks
# # =============================
#
# # Use a text splitter to break the text into manageable chunks for embedding
# text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=128)
# chunks = text_splitter.split_text(text)
#
# # =============================
# # 4. Create a Vector Store (Chroma)
# # =============================
#
# # Embed the text chunks and store them in a local Chroma vector database
# vector_store = Chroma.from_texts(chunks, embed_model)

# =============================
# 5. Create a Retriever
# =============================

# The retriever will search for relevant chunks based on user queries
retriever = vector_store.as_retriever()

# =============================
# 6. Set Up the Retrieval Chain
# =============================

# Optionally, create a simple retrieval chain (not used in final output)
# chain = create_retrieval_chain(combine_docs_chain=llm, retriever=retriever)

# =============================
# 7. Load a Retrieval-QA Prompt from LangChain Hub
# =============================

# Pull a pre-defined prompt template for retrieval-augmented QA
retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")

# =============================
# 8. Combine LLM and Prompt into a Document Chain
# =============================

# Create a chain that combines retrieved documents and the LLM using the prompt
combine_docs_chain = create_stuff_documents_chain(
    llm, retrieval_qa_chat_prompt
)

# =============================
# 9. Build the Final Retrieval Chain
# =============================

# The retrieval chain will:
#   1. Retrieve relevant chunks from the vector store
#   2. Pass them to the LLM with the prompt for answer generation
retrieval_chain = create_retrieval_chain(retriever, combine_docs_chain)

# =============================
# 10. Run the Retrieval Chain with a User Query
# =============================

# Example user query to test the RAG pipeline
# query = "Tell me name of monkeys and where do they live"
query = "How much can be converted from a traditional IRA to a Roth IRA?"
response = retrieval_chain.invoke({"input": query})

# Print the answer generated by the LLM
print(response['answer'])
