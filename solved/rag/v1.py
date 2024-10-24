"""
v1:

En general, hay dos elementos que son razonables conservar del pipeline original
de langchain porque migrarlos es más complicado y/o queremos garantizar el
mismo funcionamiento:

1. Scrapping / Doc loader
2. La estrategia de chunking / text splitting (división de texto).
   Hay algunos splitters que son más fáciles que otros. Dependiendo de la
   situación suelo reemplazarlos o no.

Podéis ver un ejemplo donde remplazo el scrapping, el doc loader y el splitter
en `v2.py`.
"""

import sys
import warnings
import bs4
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import WebBaseLoader
from langchain_chroma import Chroma
from langchain_core.documents import Document
from openai import OpenAI
from dotenv import load_dotenv

warnings.filterwarnings("ignore", message="API key must be provided")
load_dotenv()

client = OpenAI()


def llm(prompt: str, model: str = "gpt-4o-mini") -> str:
    """
    Función básica para llamar a la API. Definidla de manera que pueda ser
    sustiuible por otros proveedores en el futuro.

    Hay dos librerías que ya empaquetan una interfaz común para LLMs:
    - [`llm` de Simon Willison](https://pypi.org/project/llm/)
    - [`litellm`](https://pypi.org/project/litellm/) si no te importa usar un proxy

    Pero mi recomendación es usar la API directamente del proveedor. Cada uno
    implementa ligeras diferencias que son relevantes controlar. Además una
    librería de tercero siempre está por detrás de los cambios de API.
    """
    response = client.chat.completions.create(
        model=model, messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content


def prompt(docs: list[Document], question: str) -> str:
    """
    Prompt para usar con RAG.

    **Es MUY recomendable** descargar el prompt del hub y escribirlo en el código
    para tenerlo versionado y poder leerlo (ver slides). Por ejemplo aquí lo
    descargamos de https://smith.langchain.com/hub/rlm/rag-prompt
    """
    # Antes:
    # prompt = hub.pull("rlm/rag-prompt")

    return f"""HUMAN

You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.

Question: {question}

Context: {'\n\n'.join((doc.page_content for doc in docs))}

Answer:"""


# Mantenemos la parte de RAG, scrapeo y emebdding. Ver `v2.py`.

loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=("post-content", "post-title", "post-header")
        )
    ),
)
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,
                                               chunk_overlap=200)
splits = text_splitter.split_documents(docs)
vectorstore = Chroma.from_documents(documents=splits,
                                    embedding=OpenAIEmbeddings())

retriever = vectorstore.as_retriever()


def chatbot(question: str):
    """
    Función central de nuestro chatbot. Equivale al pipeline de LangChain.
    """
    # Obtenemos los documentos relevantes
    docs = retriever.invoke(question)

    # Generamos una respuesta
    return llm(prompt(docs, question))


if __name__ == "__main__":
    if len(sys.argv) > 1:
        question = sys.argv[1]
    else:
        question = "What is Task Decomposition?"

    print(f"Human: {question}")
    print(f"Chatbot: {chatbot(question)}")
