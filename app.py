from langchain_community.document_loaders import PyPDFLoader

from langchain.llms import HuggingFacePipeline
from transformers import LlamaTokenizer, LlamaForCausalLM, GenerationConfig, pipeline
from langchain.embeddings import HuggingFaceInstructEmbeddings
import torch 
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

from langchain_huggingface import HuggingFaceEndpoint
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
import os
from dotenv import load_dotenv
import streamlit as st


os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = " Rag Agent"

file_path = "C:\ML\Test\policy.pdf"
loader = PyPDFLoader(file_path)

docs = loader.load()

print(len(docs))

print(docs[0].page_content[0:100])
print(docs[0].metadata)
# Load environment variables
load_dotenv()
#sec_key = os.environ["HUGGINGFACEHUB_API_TOKEN"]




llm = ChatOpenAI(model="gpt-3.5-turbo-0125")
embeddings=OpenAIEmbeddings()


text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)
vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)

retriever = vectorstore.as_retriever()

system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say that you "
    "don't know. Use three sentences maximum and keep the "
    "answer concise."
    "\n\n"
    "{context}"
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)


question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)


st.title("PDF Chat Assistant")
if "history" not in st.session_state:
    st.session_state.history = []

user_question = st.text_input("Ask a question about the document:")

if user_question:
    response = rag_chain.invoke({"input": user_question})
    answer = response['answer']
    print(response["context"][0])

    # Update the chat history
    st.session_state.history.append((user_question, answer))

# Display the chat history
for i, (question, answer) in enumerate(st.session_state.history):
    st.write(f"**You:** {question}")
    st.write(f"**Assistant:** {answer}")
    st.write("---")
