"""
v5: Validación del campo `technologies`.

Ahora utilizaremos una LLM para validar el campo `technologies` con criterios
que requieres procesar lenguaje natural y razonar sobre el contenido (ver
`validate_techs`).
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


def fix_fields_prompt(
    model: Model,
    doc: str,
    parsed: dict[str, any],
    validation_errors: list[tuple[str, Exception]],
) -> str:
    return f"""Eres un experto extractor de información. Tienes que corregir los
errores de extracción ocurridos en el siguiente documento:

# Documento

{doc}

# Extracción anterior

{parsed}

# Errores de extracción

{'\n'.join(f"- {field}: {e}" for field, e in validation_errors)}

# Extracción corregida

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


def validate_output(
    output: str, model: Model
) -> tuple[dict[str, any], list[tuple[str, Exception]]]:
    """
    Parseamos y validamos la salida de la LLM.

    Devuelve una tupla con el diccionario parseado y una lista de los errores
    de validación si los hay.
    """
    parsed = parse_json_block(output)
    validation_errors = []

    for field in model.fields:
        try:
            field.validator(parsed[field.name])
        except Exception as e:
            validation_errors.append((field.name, e.args[0]))

    return parsed, validation_errors


def extractor(model: Model, doc: str, max_retries: int = 3) -> dict[str, any] | None:
    parsed, validation_errors = None, []
    for _ in range(max_retries):
        if validation_errors:
            prompt = fix_fields_prompt(model, doc, parsed, validation_errors)
        else:
            prompt = extract_fields_prompt(model, doc)

        output = llm(prompt)
        parsed, validation_errors = validate_output(output, model)
        if not validation_errors:
            return parsed
    return None


def validate_links(links: list[dict[str, str]]) -> None:
    for link in links:
        if not isinstance(link, dict):
            raise ValueError(
                "Los enlaces deben ser un diccionario con la siguiente estructura: {{'url': str, 'description': str}}"
            )
        if not link.get("url"):
            raise ValueError(f"No se ha proporcionado un enlace (`url`) en {link}")
        if not link.get("description"):
            raise ValueError(
                f"No se ha proporcionado una descripción (`description`) en {link}"
            )


def validate_techs(techs: list[str]) -> None:
    output = parse_json_block(
        llm(f"""De la siguiente lista de etiquetas: {techs} verifica que se cumplen las siguientes condiciones:
- Las siglas están descritas entre paréntesis, por ejemplo: `IPv6 (Internet Protocol Version 6)`
- Están escritas en inglés

Muestra los errores en una lista formateada como un array JSON con el siguiente formato:
                     
```json
[
    "No está en inglés: Programación en Python, ...",
    ...
]
```

Si no hay errores devuelve una lista vacía.

# Errores

```json
[""")
    )

    if output:
        raise ValueError(output)


charla = Model(
    fields=[
        Field(name="title", description="El título de la charla"),
        Field(name="speaker", description="El nombre del ponente"),
        Field(
            name="links",
            description="Los enlaces mencionados en la charla",
            validator=validate_links,
        ),
        Field(
            name="technologies",
            description="Las tecnologías mencionadas en la charla",
            validator=validate_techs,
        ),
    ]
)

if __name__ == "__main__":
    doc = get("https://pretalx.com/pycones-2024/talk/SKZFHY.ics").text
    pprint(extractor(charla, doc))
