"""
Implementaremos un extractor de información de un documento (charlas del
evento PyConES 2024) con validadores. Similar a `instructor`.

No usaremos modelos de `pydantic` para simplificar el código y para mostrar
el funcionamiento de todo el proceso.

Para ello extraeremos los campos de cada charla:

- `title`: título de la charla
- `speaker`: nombre del ponente
- `links`: enlaces mencionados en la charla
- `technologies`: tecnologías mencionadas en la charla

Deben haber dos validaciones de este modelo de datos:

- `links` debe contener 'url' y 'description' para cada enlace
- `technologies` debe estar en inglés y si hay siglas deben estar entre
   paréntesis por ejemplo: `RAG (Retrieval Augmented Generation)`

La primera validación puede hacerse programáticamente. La segunda requiere el
uso de una LLM para hacer la validación semántica.
"""
