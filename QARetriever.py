from re import template
import streamlit as st
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains.retrieval import create_retrieval_chain


def create_vector_embeddings():

    if "vectors" not in st.session_state:
        st.session_state.embeddings = OllamaEmbeddings(model = "llama3:8b")
        st.session_state.loader = PyPDFDirectoryLoader(r"C:\Users\gunda\Desktop\AI\AI-Architecture\GenAI\LangchainUpdated\Agents Fundamenals\Research paper") # Data Ingestion
        st.session_state.docs = st.session_state.loader.load() # Data Loading
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 200)
        st.session_state.final_doc = st.session_state.text_splitter.split_documents(st.session_state.docs) # Splitting Raw Data into Chunks
        st.session_state.vectors = FAISS.from_documents(st.session_state.final_doc, st.session_state.embeddings)

def generate_response(question, engine, max_tokens, temperature):


    prompt = ChatPromptTemplate.from_template(
        """
        You are a highly intelligent assistant.

        Answer ONLY from the provided context.
        If the answer is not in the context, say "I don't know".

        <context>
        {context}
        </context>

        Question: {input}
        """
        )


    model = ChatOllama(
        model = engine,
        max_tokens = max_tokens,
        temperature = temperature
        )

    doc_chain = create_stuff_documents_chain(model, prompt)
    retriever = st.session_state.vectors.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, doc_chain)
    
    answer = retrieval_chain.invoke({'input': question})
    return answer['answer']




st.title("Research Paper Assistant")

if st.button('Document Embeddings'):
    create_vector_embeddings()
    st.write("Vector Embeddings created")

query = st.text_input("Enter Your Query: ", label_visibility='collapsed')

select_model = st.sidebar.selectbox("Select Model",['llama3:8b','mistral:7b'])
temperature = st.sidebar.slider("Temperature",min_value=0.0, max_value=1.0, value = 0.2)
max_token = st.sidebar.slider("Max Tokens", min_value = 50, max_value = 300, value = 150)

if query:
    response = generate_response(query, select_model, max_token, temperature)
    st.write(response)
else:
    st.write("You are free to ask the query")



