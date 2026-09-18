from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import OllamaLLM
from langchain_classic.chains import RetrievalQA

# 1. Load the actual PDF (We removed the "with open" block!)
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader

# 1. Load all PDFs from a specific folder
folder_path = "./hr_rag_project" # Replace this with your actual folder path

# The glob parameter "**/*.pdf" tells it to look for all PDF files in the folder and any subfolders
loader = DirectoryLoader(
    folder_path, 
    glob="**/*.pdf", 
    loader_cls=PyPDFLoader
)
documents = loader.load()

print(f"\n=== LOADED {len(documents)} TOTAL PAGES/DOCUMENTS ===")
for doc in documents:
    # Get the source file name and page number from metadata
    source = doc.metadata.get('source', 'Unknown File')
    page_num = doc.metadata.get('page', 'Unknown Page')
    
    print(f"\n--- {source} (Page {page_num}) ---")
    print(doc.page_content[:200] + "...\n") # Printing just the first 200 chars for brevity
# 2. Chunk the document into smaller, searchable pieces
splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
chunks = splitter.split_documents(documents)

print("\n=== CHUNKS ===")
for i, chunk in enumerate(chunks):
    page_num = chunk.metadata.get('page', 'Unknown')
    print(f"\n--- Chunk {i+1} (From Page {page_num}) ---")
    print(chunk.page_content)
    print("-" * 40) # Adds a nice dividing line between chunks
# 3. Create Embeddings and Store them in ChromaDB
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = Chroma.from_documents(
    documents=chunks, 
    embedding=embeddings, 
    persist_directory="./chroma_db"
)



# 4. Set up the Retriever & Local LLM
retriever = vectorstore.as_retriever(search_kwargs={"k": 6})
llm = OllamaLLM(model="llama3.2") 

# 5. Build the Chain and Ask a Question!
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever
)

# Ask a real question about your document!
question1 = ["What is Grace Okafor's key strenghts", "Can you help me with the best phone in the market?", 
             "What is Rachel Kim's Goals?", "What is Hybrid in the company.", "If I had a baby, and I don't feel like working what should I do?"]

for q in question1:
    print(f"\nQuestion: {q}")
    response = qa_chain.invoke({"query": q})
    print(f"Answer: {response['result']}")