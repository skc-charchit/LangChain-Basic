import os
import streamlit as st

from doc_chat_utility import get_answer

working_dir = os.path.dirname(os.path.abspath(__file__))


st.set_page_config(
    
    page_title="Chatbot made by skc-charchit",
    page_icon="🤖",
    layout="wide"
    
)

st.title("Document Q&A with Ollama and Llama 3.2")

upload_file = st.file_uploader("Upload your file", type=["pdf"])

user_query = st.text_input("Ask your question")

if st.button("Run"):
    bytes_data = upload_file.read()
    file_name = upload_file.name
    
    file_path = os.path.join(working_dir, file_name)
    with open(file_path, "wb") as f:
        f.write(bytes_data)
    
    answer = get_answer(file_name, user_query)
    
    st.success(answer)