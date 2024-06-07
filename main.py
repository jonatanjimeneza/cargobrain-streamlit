import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAI
from langchain.document_loaders import CSVLoader
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
import os

# LLM and key loading function
def load_LLM(openai_api_key):
    """Load the language model."""
    llm = OpenAI(temperature=0, openai_api_key=openai_api_key)
    return llm

# Page title and header
st.set_page_config(page_title="Ask for CargoBrain Rates")
st.header("Ask for CargoBrain Rates")

# Input OpenAI API Key
def get_openai_api_key():
    input_text = st.text_input(
        label="OpenAI API Key",  
        placeholder="Ex: sk-2twmA8tfCb8un4...", 
        key="openai_api_key_input", 
        type="password")
    return input_text

openai_api_key = get_openai_api_key()

if openai_api_key:
    embedding = OpenAIEmbeddings(openai_api_key=openai_api_key)

    vectordb_file_path = "my_vectordb"

    def create_db():
        try:
            loader = CSVLoader(file_path='cargobrain-rates.csv')
            documents = loader.load()
            vectordb = FAISS.from_documents(documents, embedding)
            vectordb.save_local(vectordb_file_path)
            st.success("Database created successfully.")
        except Exception as e:
            st.error(f"Error creating database: {e}")

    def execute_chain():
        try:
            vectordb = FAISS.load_local(vectordb_file_path, embedding)
            retriever = vectordb.as_retriever(score_threshold=0.7)

            template = """You are a helpful assistant knowledgeable about CargoBrain rates.
            Based on the provided context from the CargoBrain rates database, answer the following question.
            If the answer is not found in the context, respond "I don't know." Do not try to make up an answer.

            CONTEXT: {context}

            QUESTION: {question}

            Instructions:
            - Answer with the specific rate if the question asks for rates (e.g., "What is the rate for 500kg from BCN to FRA?").
            - Provide information on origin, destination, and product if asked (e.g., "What products are available from MAD to MLE?").
            - If multiple rates or information are relevant, list them clearly.
            - Use information directly from the context without adding extra details.
            """

            prompt = PromptTemplate(template=template, input_variables=["context", "question"])
            
            llm = load_LLM(openai_api_key=openai_api_key)

            chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=retriever,
                input_key="query",
                return_source_documents=True,
                prompt=prompt
            )

            return chain
        except Exception as e:
            st.error(f"Error executing chain: {e}")
            return None

    if __name__ == "__main__":
        create_db()

    btn = st.button("Re-create database")
    if btn:
        create_db()

    question = st.text_input("Question: ")

    if question:
        chain = execute_chain()
        if chain:
            response = chain({"query": question})
            st.header("Answer")
            st.write(response['result'])
            st.write("Source documents:")
            for doc in response['source_documents']:
                st.write(doc.page_content)
