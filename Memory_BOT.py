from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_ollama import ChatOllama
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_community.embeddings import OllamaEmbeddings
import streamlit as st
from langchain_classic.chains.history_aware_retriever import create_history_aware_retriever
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory


#----------------------------------
# VECTOR STORE CREATION                  
#----------------------------------

def create_embeddings_vectorstore():

    if "vectors" not in st.session_state:
    
        # Data Ingestion
        st.session_state.loader = PyPDFDirectoryLoader(r"C:\Users\gunda\Desktop\AI\AI-Architecture\GenAI\LangchainUpdated\Agents Fundamenals\Research paper")
        # Loading the Data
        st.session_state.raw_data = st.session_state.loader.load()
        # Calling Embedding Model
        st.session_state.embeddings = OllamaEmbeddings(model = "nomic-embed-text:latest")
        # Breaking Raw data into chunks
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap = 100)
        # Creating Chunks (Documents)
        st.session_state.docs = st.session_state.text_splitter.split_documents(st.session_state.raw_data)
        # Embedding the Chunks and Storing in Vector Database FAISS
        st.session_state.vectors = FAISS.from_documents(st.session_state.docs, st.session_state.embeddings)

#----------------------------------
# HISTORY AWARE RETRIEVER            
#----------------------------------

def rephrase_retrieval(engine):

    system_prompt = """Given a chat history and the latest user question \
        which might reference context in the chat history, formulate a standalone question \
        which can be understood without the chat history. Do NOT answer the question, \
        just reformulate it if needed and otherwise return it as is."""
    
    prompt = ChatPromptTemplate.from_messages(
        [
            ('system',system_prompt),
            MessagesPlaceholder('chat_history'),
            ('human','{input}' )
        ]
    )
    history_aware_retrieve = create_history_aware_retriever(llm = engine, retriever = st.session_state.vectors.as_retriever(), prompt = prompt)
    
    return history_aware_retrieve

#----------------------------------
# RESPONSE FUNCTION               
#----------------------------------

def generate_response(question, engine, temperature, max_tokens):


    qa_system_ = """
        You are a highly intelligent assistant.

        Answer ONLY from the provided context.
        If the answer is not in the context, say "I don't know".

        <context>
        {context}
        </context>

        Question: {input}
        """
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ('system',qa_system_),
            MessagesPlaceholder('chat_history'),
            ('human',"{input}")
        ]
    )
    
    model = ChatOllama(model = engine, model_provider = "ollama", temperature=temperature, max_tokens = max_tokens)

    doc_chain = create_stuff_documents_chain(model,qa_prompt)
    history_aware_chain = rephrase_retrieval(model)
    retriever_chain = create_retrieval_chain(history_aware_chain, doc_chain)

    #----------------------------------
    # MEMORY MANAGEMENT                   
    #----------------------------------

    if 'store' not in st.session_state:
        st.session_state.store = {}
    
    def get_session_history(session_id:str):
        if session_id not in st.session_state.store:
            st.session_state.store[session_id] = ChatMessageHistory()
        return st.session_state.store[session_id]

    rag_chain = RunnableWithMessageHistory(
        retriever_chain, get_session_history,
        input_messages_key= "input",
        history_messages_key="chat_history",
        output_messages_key="answer"
    )
    response = rag_chain.invoke(
        {"input": question},
        config={"configurable": {"session_id": "default"}}
    )

    return response["answer"]

    
        
    
#----------------------------------
# STREAMLIT UI                    
#----------------------------------


st.title("Hello Ji")

if st.button('Document Embeddings'):
    create_embeddings_vectorstore()
    st.success("Vectors stored in FAISS")

query = st.text_input("Enter Your Query", label_visibility="collapsed")

select_model = st.sidebar.selectbox("Select Model",['mistral:7b','phi3:mini'])
temperature = st.sidebar.slider("Temperature",min_value=0.0, max_value=1.0, value = 0.2)
max_token = st.sidebar.slider("Max Tokens", min_value = 50, max_value = 300, value = 150)

if query:
    if "vectors" not in st.session_state:
        st.warning("Please click 'Document Embeddings' first")
    else:
        answer = generate_response(
            query, select_model, temperature, max_token
        )
        st.write(answer)