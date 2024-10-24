# GenAI ❤️ f-string: Desarrollando con IA Generativa sin cajas negras

[English version here](https://github.com/double-thinker/genai_loves_fstrings_english)

[Slides](https://raw.githubusercontent.com/double-thinker/genai_loves_fstrings/main/slides/talk.pdf)

## TLDR

Los frameworks de IA Generativa generan "framework-lock" muy rápido y en ocasiones
no merece la pena. Se puede reimplementar en python de forma más legible,
debugeable y mantenible. En mi experiencias en consultoría veo como la deuda
técnica y el "lock-in" supera a las ventajas de los frameworks.

En este taller veremos a través de ejemplos cómo se puede implementar pipelines
moderadamente complejos sin usar frameworks entendiendo sus ventajas e
inconvenientes.



[Taller impartido en la PyConES 2024](https://pretalx.com/pycones-2024/talk/SKZFHY/).

## Descripción

Los _frameworks_ de IA Generativa son demasiado "mágicos": ocultan detalles en
pipelines aparentemente complejos. Operaciones simples como interpolaciones de
texto (un `f-string`, vaya) se llaman `StringConcatenationManagerDeluxe`
(dramatización).

Estas herramientas aceleran el desarrollo en las fases iniciales pero a un
precio: el mantenimiento, observabilidad y la independencia se resienten. Cada
vez hay más voces en la comunidad ([[1]](https://hamel.dev/blog/posts/prompt/),
[[2]](https://github.com/langchain-ai/langchain/discussions/18876)) que
comienzan a explicitar este problema.

Este taller pretende resumir mi experiencia desarrollando productos con LLMs
para evitar el "_framework-lock_" y hacer desarrollos más fáciles de extender,
migrar y debuguear:

- Parte 1: "Framework-lock": ¿Merece la pena? ¿Cómo migrar?
- Parte 2: Show me the prompt: destripando frameworks actuales
- Parte 3: Desde cero: mejores prácticas para desarrollar aplicaciones complejas

Si estás desarrollando aplicaciones con LLMs y notas que te "peleas" con tu
_framework_ o si vas a empezar a desarrollar con IA Generativa y quieres
ahorrarte bastantes quebraderos de cabeza este taller te puede interesar.

## Instrucciones

### Entorno de desarrollo

Para realizar el taller puedes usar el devcontainer ya configurado en GitHub Codespaces sin
instalar nada más (recomendado):

[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/double-thinker/genai_loves_fstrings/?quickstart=1)

Para activar el entorno virtual y ejecutar los ejemplos:

```bash
source .venv/bin/activate
# python -m ...
```

Si prefieres usar tu editor favorito puedes clonar el repositorio y usar `uv` para instalar el entorno:

```bash
git clone https://github.com/double-thinker/genai_loves_fstrings
cd genai_loves_fstrings
uv sync
source .venv/bin/activate
```

### Cuenta de OpenAI

Añade tu API key al fichero `.env`. Tienes un ejemplo en `.env.example`.
Necesitarás entre 1 y 3€ en créditos para completar el taller.

## Actividades

Aunque puedes ver el resultado de las actividades en `solved/`, te recomiendo
que leas en paralelo la presentación de la [charla](slides/talk.pdf) y hagas las actividades
paso a paso.

Las soluciones tienen iteraciones que puedes ir viendo en los ficheros `v1.py`,
`v2.py`, etc.

### 01. Ejemplo de RAG

RAG es considerado el "Hello world" de la IA Generativa. Tomando un ejemplo de
RAG de la documentación de LangChain vamos a ver las limitaciones y
reimplementarlo de manera que sea más legible y fácil de mantener.

En `rag.py` encontrarás el ejemplo de RAG sin modificaciones y en
`solved/rag` la solución.

```bash
python -m rag

# Soluciones
python -m solved.rag.v1
python -m solved.rag.v2
```

Ejemplo:

```
Human: What is Task Decomposition?
Chatbot: Task Decomposition is the process of breaking down complex tasks into smaller, more manageable steps. This can be achieved through techniques like Chain of Thought (CoT) prompting, which encourages the model to think step by step, and Tree of Thoughts, which explores multiple reasoning possibilities at each step. The decomposition can be guided by simple prompts, task-specific instructions, or human input.
```

Puedes probar con diferentes preguntas:

```bash
python -m rag "What is RAG?"
python -m rag "What is a chatbot?"
```

### 02. `SmartLLMChain`

Ahora analizaremos `SmartLLMChain` interceptando las llamadas a la API de OpenAI
como se sugiere en el [artículo de Hamel](https://hamel.dev/blog/posts/prompt/)
pero sin necesitar de un proxy que intercepte las llamadas como `mitmproxy`.

Para ello se proporciona el módulo `observability` que muestra en la terminal
el contenido de las llamadas a la API de OpenAI y su resultado.

Conociendo su funcionamiento reimplementaremos `SmartLLMChain` sin usar
`langchain` para comparar las ventajas e inconvenientes de ambas aproximaciones.

Solución:

```bash
python -m solved.smartllm.v2 "What is the meaning of life?"
```

```
The researcher thought that the flaws in Idea 1 were less significant compared to Idea 2.

Improved Answer:
The meaning of life is a complex and deeply philosophical question that has been debated for centuries. While different individuals and cultures may have unique perspectives on the purpose of life, it is essential to consider both subjective beliefs and objective truths in exploring this topic. One possible perspective is that the meaning of life is to seek happiness, fulfillment, and personal growth, while others may find purpose in serving others, making a positive impact on society, or pursuing spiritual enlightenment. It is crucial to reflect on one's values, goals, and aspirations to find a sense of purpose and direction in life. Ultimately, living a meaningful and intentional life can lead to a sense of fulfillment and satisfaction, regardless of external circumstances.
```

### 03. Extracción de información con validadores

Por último realizaremos un caso de extracción de información de documentos. Hay
varias maneras de hacerlo: utilizando la API de un proveedor que soporte llamada
de funciones (por ejemplo, Anthropic u OpenAI), utilizando [búsqueda restringida](https://huggingface.co/blog/constrained-beam-search)
que requiere control sobre cómo se realiza la inferencia (algunos proveedores
no lo soportan) o utilizando prompt engineering y una lógica de validación.

Esta última estrategia es la que usaremos para mostrar cómo se puede implementar
flujos más complejos con python directamente sin necesidad de frameworks.

```bash
python -m solved.extractor.v5
```

Genera un diccionario con los datos extraídos del documento:

```python
{'links': [{'description': 'Discusión sobre el problema de los frameworks de '
                           'IA',
            'url': 'https://hamel.dev/blog/posts/prompt/'},
           {'description': 'Discusión sobre el framework-lock',
            'url': 'https://github.com/langchain-ai/langchain/discussions/18876'},
           {'description': 'Tutorial introductorio sobre LLM usando langchain',
            'url': 'https://python.langchain.com/v0.1/docs/use_cases/question_answering/'}],
 'speaker': 'Alejandro Vidal',
 'technologies': ['OpenAI', 'LLMs', 'langchain'],
 'title': 'GenAI❤️f-string. Desarrollando con IA Generativa sin cajas negras.'}
```

## Importante

- No vamos a usar las mejores prácticas para construir los prompts. El objetivo
  es comparar implementaciones desde cero vs frameworks.
- De igual manera la estructura del código está diseñada para la docencia
  durante el taller y no es la más adecuada para un desarrollo productivo.
