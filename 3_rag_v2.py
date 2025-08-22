import os
from dotenv import load_dotenv

from langsmith import traceable # <---- Key Import

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

os.environ['LANGCHAIN_PROJECT'] = "Rag App v2"


config = {
    "run_name":"pdf_rag_query"
}

load_dotenv()  # expects OPENAI_API_KEY in .env
PDF_PATH = r"C:\Users\Shubham\Downloads\IOT Notes (1).pdf"

# 1) Load PDF
@traceable(name='load_pdf', tags=['pdf','loader'])
def load_pdf(path: str):
    loader = PyPDFLoader(path)
    return loader.load()  # one Document per page
    

@traceable(name='split_documents')
def split_documents(docs,chunk_size=1000,chunk_overlap=150):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    splits = splitter.split_documents(docs)

    return splits

@traceable(name="build_vectorstore")
def build_vectorstore(splits):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    vs = FAISS.from_documents(splits, embeddings)
    return vs


@traceable(name='setup_pipeline')
def setup_pipeline(pdf_path: str):
    docs = load_pdf(pdf_path)
    splits = split_documents(docs)
    vs = build_vectorstore(splits)
    return vs


prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer ONLY from the provided context. If not found, say you don't know."),
    ("human", "Question: {question}\n\nContext:\n{context}")
])

llm = ChatGroq(model="llama3-70b-8192")
def format_docs(docs): return "\n\n".join(d.page_content for d in docs)

vectorstore = setup_pipeline(PDF_PATH)
retriever = vectorstore.as_retriever(search_type="similarity",search_kwargs={"k":4})

parallel = RunnableParallel({
    "context": retriever | RunnableLambda(format_docs),
    "question": RunnablePassthrough()
})

chain = parallel | prompt | llm | StrOutputParser()

# 6) Ask questions
print("PDF RAG ready. Ask a question (or Ctrl+C to exit).")
q = input("\nQ: ")
ans = chain.invoke(q.strip(),config=config)
print("\nA:", ans)