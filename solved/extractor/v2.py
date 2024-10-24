"""
v2: Añadimos el parser de bloques de JSON y procesamos un primer documento
(esta charla): https://pretalx.com/pycones-2024/talk/SKZFHY.ics
"""

import json
import re
from dataclasses import dataclass
from pprint import pprint
from typing import Callable

from dotenv import load_dotenv
from openai import OpenAI
from requests import get

load_dotenv()
client = OpenAI()


def llm(prompt: str, model: str = "gpt-4o-mini") -> str:
    response = client.chat.completions.create(
        model=model, messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content


@dataclass
class Field:
    """
    Un campo a extraer de un documento.

    `validator` es una función que valida el campo y lanza una excepción si no
    es válido.
    """

    name: str
    description: str
    validator: Callable[[any], None] = lambda _: None

    def __str__(self) -> str:
        return f"'{self.name}': '{self.description}'"


@dataclass
class Model:
    fields: list[Field]


charla = Model(
    fields=[
        Field(name="title", description="El título de la charla"),
        Field(name="speaker", description="El nombre del ponente"),
        Field(name="links", description="Los enlaces mencionados en la charla"),
        Field(
            name="technologies", description="Las tecnologías mencionadas en la charla"
        ),
    ]
)


def extract_fields_prompt(model: Model, doc: str) -> dict[str, any]:
    """
    Genera un prompt (quizá excesivamente irónico) para extraer los campos de
    un documento.
    """
    return f"""Eres un experto extractor de información. De pequeño soñaste con
este trabajo. Ahora puedes hacerlo realidad. De ello depende el futuro de
la humanidad. Además si lo haces bien tendrás una propina de 100k€.

# Documento

{doc}


# Campos

Campos a extraer: {model.fields}.

```json
"""


JSON_BLOCK_RE = re.compile(r"```json\s*([\s\S]*)\s*```")


def parse_json_block(output: str) -> dict[str, any]:
    """
    Procesamos la salida de la LLM buscando un bloque JSON de markdown.

    Truco: usar un formato como markdown facilita el parseo de la respuesta.
    Las LLMs están entrenadas con muchos documentos MD y trabajan bien con esa
    estructura y sintáxis (document mimicking).
    """
    match = JSON_BLOCK_RE.search(output)
    if not match:
        raise ValueError("No se ha encontrado un bloque JSON")
    return json.loads(match.group(1))


def extractor(model: Model, doc: str) -> dict[str, any] | None:
    output = llm(extract_fields_prompt(model, doc))
    return parse_json_block(output)


if __name__ == "__main__":
    doc = get("https://pretalx.com/pycones-2024/talk/SKZFHY.ics").text
    pprint(extractor(charla, doc))
