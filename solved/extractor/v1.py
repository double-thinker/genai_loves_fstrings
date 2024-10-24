"""
v1: creamos un modelo de datos para las charlas y los prompts para extraer
campos de un documento.

Podríamos usar pydantic para definir el modelo de datos. Para hacer visible
todo el proceso haremos a mano un gestor de modelos básico (`Model` y
`Field`).
"""

from dataclasses import dataclass
from typing import Callable

from dotenv import load_dotenv
from openai import OpenAI

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
