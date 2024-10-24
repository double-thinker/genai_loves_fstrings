"""
Ejercicio 1:
Ejemplo tomando de https://python.langchain.com/docs/tutorials/rag/ con algunos
cambios menores (dotenv vs hardcoded API key, CLI e ignorar warnings de LangSmith)

Reimplementaremos este pipeline usando "python solo". Usaremos chroma como
almacén de vectores. El text splitter y el web base loader están permitidos.
Puedes usar otras librerías como beautifulsoup, requests, etc.

Sugerencias:
- Comienza con una solución simple: contexto del rag ficticio, sin usar un 
  vector store
- Haz una función para llamadas a la API de OpenAI sencilla
- Haz otra función para crear el prompt

La solución está en el módulo `solved`
"""

import sys
import warnings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import WebBaseLoader
from langchain_chroma import Chroma
from langchain import hub
from langchain_openai import ChatOpenAI

import bs4
import dotenv

warnings.filterwarnings("ignore", message="API key must be provided")

dotenv.load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini")

# Load, chunk and index the contents of the blog.
loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=("post-content", "post-title", "post-header")
        )
    ),
)
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)
vectorstore = Chroma.from_documents(
    documents=splits, embedding=OpenAIEmbeddings())

# Retrieve and generate using the relevant snippets of the blog.
retriever = vectorstore.as_retriever()
prompt = hub.pull("rlm/rag-prompt")


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        question = sys.argv[1]
    else:
        question = "What is Task Decomposition?"

    print(f"Human: {question}")
    print(f"Chatbot: {rag_chain.invoke(question)}")
