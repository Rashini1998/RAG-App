import os
from langchain_community.document_loaders import TextLoader, DirectoryLoader #to read the text files form a particular directory
from langchain_text_splitters import CharacterTextSplitter # to chunk the text files into smaller pieces
#from langchain_openai import OpenAIEmbeddings # to convert the text chunks into vector embeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma #to store the vector embeddings in a vector database
from dotenv import load_dotenv # to load environment variables from a .env file

load_dotenv()

#01
# to load all the text files from the docs directory
def load_documents(docs_path="docs"):
    """Load all text files from the docs directory"""
    print(f"Loading documents from {docs_path}...")
    
    # Check if docs directory exists
    if not os.path.exists(docs_path):
        raise FileNotFoundError(f"The directory {docs_path} does not exist. Please create it and add your company files.")
    
    # Load all .txt files from the docs directory
    loader = DirectoryLoader(
        path=docs_path,
        glob="*.txt",
        loader_cls=TextLoader, #beacuse we are loading text files, we use TextLoader class
        loader_kwargs={"encoding": "utf-8"}
    )
    
    documents = loader.load()
    
    if len(documents) == 0:
        raise FileNotFoundError(f"No .txt files found in {docs_path}. Please add your company documents.")
    
   
    for i, doc in enumerate(documents[:2]):  # Show first 2 documents
        print(f"\nDocument {i+1}:")
        print(f"  Source: {doc.metadata['source']}")
        print(f"  Content length: {len(doc.page_content)} characters")
        print(f"  Content preview: {doc.page_content[:100]}...")
        print(f"  metadata: {doc.metadata}")

    return documents

#02
# to split the documents into smaller chunks with overlap
def split_documents(documents, chunk_size=800, chunk_overlap=0):
    """Split documents into smaller chunks with overlap"""
    print("Splitting documents into chunks...")
    
    text_splitter = CharacterTextSplitter(
        chunk_size=chunk_size, 
        chunk_overlap=chunk_overlap
    )
    
    chunks = text_splitter.split_documents(documents)
    
    if chunks:
    
        for i, chunk in enumerate(chunks[:5]):
            print(f"\n--- Chunk {i+1} ---")
            print(f"Source: {chunk.metadata['source']}")
            print(f"Length: {len(chunk.page_content)} characters")
            print(f"Content:")
            print(chunk.page_content)
            print("-" * 50)
        
        if len(chunks) > 5:
            print(f"\n... and {len(chunks) - 5} more chunks")
    
    return chunks

#03
# to create and persist ChromaDB vector store
def create_vector_store(chunks, persist_directory="db/chroma_db"):
    """Create and persist ChromaDB vector store"""
    print("Creating embeddings and storing in ChromaDB...")

    embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

   # to initialize the OpenAI embeddings model     
   #embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
    
    # Create ChromaDB vector store
    print("--- Creating vector store ---")
    vectorstore = Chroma.from_documents(
        documents=chunks, # to take all the chunks
         embedding=embeddings,
       #embedding=embedding_model, # to convert the chunks into vector embeddings   
        persist_directory=persist_directory, # to specify the directory where the vector store will be persisted
        collection_metadata={"hnsw:space": "cosine"} # to specify the distance metric for the vector store (algorithm used for nearest neighbor search)
    )
    print("--- Finished creating vector store ---")
    
    print(f"Vector store created and saved to {persist_directory}")
    return vectorstore

def main():
    print("Main Function")

    #1. Loading the files
    documents = load_documents(docs_path="docs")

    #2. Chuncking the files
    chunks = split_documents(documents)

    #3. Embedding and Storing in Vector DB
    vectorstore = create_vector_store(chunks)

if __name__ == "__main__":
    main()
