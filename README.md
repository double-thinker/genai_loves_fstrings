# GenAI ❤️ f-string: Desarrollando con IA Generativa sin cajas negras

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

- Parte 1: Show me the prompt: destripando frameworks actuales
- Parte 2: "Framework-lock": ¿Merece la pena? ¿Cómo migrar?
- Parte 3: Desde cero: mejores prácticas para desarrollar aplicaciones complejas

Si estás desarrollando aplicaciones con LLMs y notas que te "peleas" con tu
_framework_ o si vas a empezar a desarrollar con IA Generativa y quieres
ahorrarte bastantes quebraderos de cabeza este taller te puede interesar.

Para realizar el taller se necesitan conocimientos básicos de LLM: saber
utilizar una API de LLMs y conocer el funcionamiento de un RAG (por ejemplo,
[este tutorial es una introducción usando langchain](https://python.langchain.com/v0.1/docs/use_cases/question_answering/)).

**Los ejercicios del taller utilizarán OpenAI como servicio de inferencia. Los
participantes deben tener una cuenta activa con créditos suficientes (aprox.
1-3€)**
