import os
import pandas as pd
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import DataFrameLoader

# Get Key from OPENAPI
os.environ["OPENAI_API_KEY"] = "xxxxxxxxxx"

print("Loading and cleaning dataset...")
df = pd.read_csv("single_qna.csv")

# Search for column that contains the "Answer" text
target_column = 'Answer' 
df = df.dropna(subset=[target_column])
df[target_column] = df[target_column].astype(str)

# Convertnthe information to a document
loader = DataFrameLoader(df, page_content_column=target_column)
documents = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
docs = text_splitter.split_documents(documents)

# Store the information in a vector
print("Creating vector database... this may take a moment.")
embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(docs, embeddings)

# Obtain the QA Chain
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever()
)

# Interface
print("-" * 30)
print("AI Chatbot is operational! Type 'exit' to stop.")
print("-" * 30)

while True:
    user_input = input("You: ")
    if user_input.lower() in ['exit', 'quit']:
        break
    
    try:
        response = qa_chain.invoke(user_input)
        print(f"\nBot: {response['result']}\n")
    except Exception as e:
        print(f"error: {e}")