import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# get_pdf_text(pdf_docs)
# This function takes a list of PDF files as input and returns the text content extracted from all the PDF files as a single string.
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# get_text_chunks(text)
# This function takes the text content as input and splits it into smaller chunks of text using the RecursiveCharacterTextSplitter from LangChain. The chunk size is set to 10,000 characters with an overlap of 1,000 characters.
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

# get_vector_store(text_chunks)
# This function creates a vector store using the FAISS vector store from LangChain. It takes the text chunks as input, embeds them using the GoogleGenerativeAIEmbeddings, and saves the vector store locally as "faiss_index".
def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

# get_conversational_chain()
# This function sets up a conversational chain using the ChatGoogleGenerativeAI model from LangChain. It defines a prompt template that instructs the model to generate three types of questions and their corresponding answers based on the given context: True or False questions, Multiple Choice Questions (MCQs), and One-word answer questions. The chain is then loaded and returned.
def get_conversational_chain():
    prompt_template = """
Based on the given text, generate the following types of questions and their corresponding answers:
1. True or False questions (5 questions and thier answers)
2. Multiple Choice Questions (MCQs) with 4 choices each (5 questions and thier answers)
3. One-word answer questions (5 questions and thier answers)

Text: {context}
"""
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

# user_input(user_question)
# This function handles the user's question input. It loads the vector store from the local "faiss_index" file, performs a similarity search against the user's question, and generates a response using the conversational chain created by get_conversational_chain(). The response is then printed and displayed in the Streamlit app.
def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    print(response)
    st.write("Reply: ", response["output_text"])




# main()
# This is the main function that sets up the Streamlit app. It configures the page title and header, and creates a sidebar for uploading PDF files. When the "Submit & Process" button is clicked, it extracts the text from the uploaded PDFs, splits the text into chunks, and creates the vector store. The user can then input a question (or use the default "Create Question Answers" prompt), and the app will generate and display the response.
def main():
    st.set_page_config("Chat PDF")
    st.header("Generate Questions and Answers from PDF using Gemini💁")
    user_question = "Create Questions Answers with proper indentation"
    if st.button("Generate Q&A"):
        user_input(user_question)
    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
            st.success("Done")

if __name__ == "__main__":
    main()
