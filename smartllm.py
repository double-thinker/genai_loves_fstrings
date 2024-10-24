"""
Ejercicio 2:
Reimplementa el algoritmo de smartllm usando langchain. Para ello interceptaremos
las llamadas a la API de OpenAI usando el módulo `observability` que hay
en este repositorio.
"""

import sys

from langchain.prompts import PromptTemplate
from langchain_experimental.smart_llm import SmartLLMChain
from langchain_openai import ChatOpenAI

DEFAULT = """¿Cuál es el sentido de la vida?"""


def smartllm(pregunta: str = DEFAULT):
    prompt = PromptTemplate.from_template(pregunta)
    llm = ChatOpenAI(temperature=0.1, model_name="gpt-3.5-turbo")

    chain = SmartLLMChain(llm=llm, prompt=prompt, n_ideas=2, verbose=False)
    return chain.invoke({})["resolution"]


if __name__ == "__main__":
    question = sys.argv[1] if len(sys.argv) > 1 else DEFAULT
    print(smartllm(question))
