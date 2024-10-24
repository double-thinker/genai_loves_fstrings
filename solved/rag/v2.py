"""
v2:

Cambios:

- Creamos `text_splitter` para dividir los documentos en chunks (comportamiento
ligeramente diferente a `RecursiveCharacterTextSplitter`, IMO mejor). Esta es
la parte más extensa de esta versión y no recomiendo hacerlo como primer paso
cuando haces una migración desde langchain. Pero se puede reutilizar el código
y es muy sencillo de entender
- Usamos ChromaDB directamente sin langchain
- Creamos `scrape_web` para obtener el contenido de la web


Extra:
- Persistimos la BBDD para evitar costes de creación de la colección en cada
  ejecución
"""

import os
import re
import sys
from typing import Generator
from uuid import uuid4

import bs4

# Nota: No usamos `langchain_chroma` sino `chromadb`
import chromadb
import chromadb.utils.embedding_functions as ef
import requests
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI()
db = chromadb.PersistentClient(path="./ragdatabase")


def llm(prompt: str, model: str = "gpt-4o-mini") -> str:
    response = client.chat.completions.create(
        model=model, messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content


def prompt(docs: list[str], question: str) -> str:
    return f"""HUMAN

You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.

Question: {question}

Context: {'\n\n'.join(docs)}

Answer:"""


def scrape_web(url: str) -> str:
    response = requests.get(url)
    soup = bs4.BeautifulSoup(response.text, "html.parser")
    elements = [soup.find(class_="post-title"), soup.find(class_="post-content")]
    return " ".join([element.get_text() for element in elements])


def text_splitter(doc: str, chunk_size: int = 1000, chunk_overlap: int = 200):
    """
    Simple splitter similar a `RecursiveCharacterTextSplitter` de langchain.
    Con leves diferencias: los caracteres de división son más consistentes.

    A pesar de su aparente "complejidad", este es un código que normalmente
    reutilizo o si es necesario para el proyecto mantengo el text splitter de
    langchain.
    """
    splits = re.split(r"[\s\.,;:]+", doc)

    # Acumulamos los splits hasta alcanzar el chunk size
    # añadiento un overlap de `chunk_overlap` caracteres del chunk anterior
    prev_chunk = ""
    chunk = ""
    for subchunk in splits:
        length = len(chunk) + len(subchunk)
        if length > chunk_size:
            yield prev_chunk[-chunk_overlap:] + chunk
            prev_chunk = chunk
            chunk = ""
        else:
            chunk += " " + subchunk
    if chunk:
        yield chunk


def fill_db(docs: Generator[str, None, None]):
    """
    Función para llenar la base de datos con los documentos y rellenarla con
    chunks de los documentos.

    Para mí el manejo de la BBDD directamente es algo que me ayuda mucho a
    debuggear y controlar el pipeline: cacheo, reutilización, uso de varios
    embeddings, etc.
    """
    collection = db.get_or_create_collection(
        "rag",
        embedding_function=ef.OpenAIEmbeddingFunction(
            model_name="text-embedding-ada-002", api_key=os.getenv("OPENAI_API_KEY")
        ),
    )

    if collection.count() > 0:
        # Si ya existe la base de datos, se salta la creación para ahorrar
        # tiempo (y dinero).
        return collection

    for doc in docs:
        for chunk in text_splitter(doc):
            collection.add(documents=chunk, ids=[uuid4().hex])

    return collection


def chatbot(question: str):
    """
    Función central de nuestro chatbot. Equivale al pipeline de LangChain.
    """
    db = fill_db([scrape_web("https://lilianweng.github.io/posts/2023-06-23-agent/")])
    context = db.query(query_texts=[question], n_results=5)["documents"][0]
    return llm(prompt(context, question))


if __name__ == "__main__":
    if len(sys.argv) > 1:
        question = sys.argv[1]
    else:
        question = "What is Task Decomposition?"

    print(f"Human: {question}")
    print(f"Chatbot: {chatbot(question)}")
