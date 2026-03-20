import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama

prompt = ChatPromptTemplate.from_messages(
    [
        ("system","You are helpful assistant with 180+IQ"),
        ("user","{question}")
    ]
)

def generate_response(question, engine, max_tokens, temperature):
    model = ChatOllama(
        model = engine,
        max_tokens = max_tokens,
        temperature = temperature
        )
    output_parser = StrOutputParser()
    chain = prompt | model | output_parser
    answer = chain.invoke({'question':question})
    return answer

st.title("Q&A ChatBot")

st.sidebar.title("Settings")

select_model = st.sidebar.selectbox("Select Model",['llama3:8b','mistral:7b'])

temperature = st.sidebar.slider("Temperature",min_value=0.0, max_value=1.0, value = 0.2)
max_token = st.sidebar.slider("Max Tokens", min_value = 50, max_value = 300, value = 150)

# Main Interface for the user
st.write("Hello Ji !!!")

user_input = st.text_input("You:", label_visibility="collapsed")

if user_input:
    response = generate_response(user_input, select_model, max_token, temperature)
    st.write(response)

else:
    st.write("You are free to ask any query")