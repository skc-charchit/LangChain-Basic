import os
from langchain_community.llms import Ollama
from langchain_community.document_loaders import UnstructuredFileLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA

# Set working directory
working_dir = os.path.dirname(os.path.abspath(__file__))

# Initialize the LLM
llm = Ollama(
    model="llama3.2:latest",
    temperature=0
)

# Initialize embeddings
embeddings = HuggingFaceEmbeddings()

def get_answer(file_name, query):
    # Construct the file path using os.path.join for better compatibility
    file_path = os.path.join(working_dir, '..', 'data', file_name)

    # Load documents from the specified file, handling potential exceptions
    try:
        loader = UnstructuredFileLoader(file_path)
        documents = loader.load()
    except FileNotFoundError:
        raise FileNotFoundError(f"No such file: {file_path}")
    except Exception as e:
        print(f"An error occurred while loading documents: {e}")
        return "Error loading documents."

    # Create text chunks from documents using RecursiveCharacterTextSplitter
    text_splitter = CharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""]
    )
    text_chunks = text_splitter.split_documents(documents)

    # Create a knowledge base using FAISS vector store
    knowledge_base = FAISS.from_documents(text_chunks, embeddings)

    # Initialize Retrieval QA chain with the LLM and retriever from knowledge base
    qa = RetrievalQA.from_chain_type(llm=llm, retriever=knowledge_base.as_retriever())

    # Invoke the QA chain with the user query and return the result
    try:
        response = qa.invoke({"query": query})
        return response["result"]
    except Exception as e:
        print(f"An error occurred while processing the query: {e}")
        return "Error processing query."




