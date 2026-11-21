# Clasificación supervisada de relevancia de memorias en conversaciones de largo plazo

Proyecto final del módulo de Aprendizaje Supervisado · Maestría en Inteligencia Artificial, USFQ

**Autor:** Roberth Chachalo

## El problema

Un agente conversacional que acompaña a una persona durante meses acumula cientos de *memorias*. Ante una pregunta nueva debe decidir cuáles traer. La búsqueda por similitud de embeddings devuelve turnos parecidos a la pregunta, pero muchos no contienen la respuesta.

Este proyecto entrena un clasificador binario que, dado un par *(pregunta, memoria candidata)*, predice si la memoria contiene evidencia para responder. Se compara contra la línea base de ordenar los candidatos solo por similitud coseno.

Todo el trabajo está en un único notebook: [`relevancia_memorias_locomo.ipynb`](relevancia_memorias_locomo.ipynb).

## Datos

Se usa [LoCoMo](https://github.com/snap-research/locomo) (Maharana *et al.*, ACL 2024): 10 conversaciones largas y 1.986 preguntas con la evidencia anotada por humanos.

1. Se toma una muestra de 500 preguntas (sin la categoría adversarial).
2. Cada pregunta y cada turno de su conversación se convierten en embeddings con Qwen3-Embedding-0.6B.
3. Por pregunta se conservan los 20 turnos más similares.
4. Cada par recibe la etiqueta 1 si el turno está en la evidencia y 0 si no, y se describe con 14 features semánticas, léxicas, temporales y de contexto.

El resultado es una tabla de 10.000 pares con 5,1% de positivos.

## Contenido del notebook

| Sección | Contenido |
|---|---|
| 1 | Introducción y motivación |
| 2 | Los datos: LoCoMo |
| 3 | Extracción de características con un modelo de embeddings preentrenado (*transfer learning*) |
| 4 | Construcción de la tabla de pares y de las 14 features |
| 5 | Separación de los datos sin fuga de información (`StratifiedGroupKFold` agrupado por pregunta) |
| 6 | Análisis exploratorio: balance de clases, distribuciones, correlaciones y recall del *retrieval* |
| 7 | Visualización del espacio de features con t-SNE |
| 8 | Pipelines, selección de features por información mutua, búsqueda de hiperparámetros con *k-fold* repetido (10 × 10), curvas de validación y de aprendizaje |
| 9 | Comparación estadística de regresión logística y Random Forest (test de Wilcoxon y test t corregido) |
| 10 | Conclusiones |

## Resultados

PR-AUC (*average precision*) en 100 particiones (10 folds × 10 repeticiones). Un clasificador al azar obtendría 0,051.

| Modelo | PR-AUC medio | Desviación estándar |
|---|---|---|
| Línea base: `sim_coseno` | 0,176 | 0,055 |
| Línea base: `sim_z` (similitud normalizada por pregunta) | 0,325 | 0,059 |
| Random Forest | 0,420 | 0,067 |
| **Regresión logística** | **0,442** | 0,062 |

- Los dos modelos superan a la similitud sola de forma significativa (p < 0,001).
- La regresión logística supera al Random Forest por 0,022 puntos; el test de Wilcoxon rechaza la igualdad, pero el test t corregido de Nadeau y Bengio no (p ≈ 0,095).
- En el 16,8% de las preguntas la evidencia no llega a los 20 candidatos: el *retrieval* es el techo del sistema.

## Cómo ejecutarlo

### En Google Colab

1. Abrir el notebook en Colab: [![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/betochalo/relevancia-memorias-locomo/blob/main/relevancia_memorias_locomo.ipynb)
2. En la primera celda de código, descomentar las tres líneas indicadas: clonan este repositorio (con los datos ya calculados) e instalan `rank-bm25`.
3. Ejecutar todas las celdas.

En Colab no hace falta Ollama: los embeddings se leen de `data/embeddings.npz`.

La búsqueda de hiperparámetros usa 100 particiones y es la parte más lenta. Para acortarla, cambiar `K_FOLDS = 5` y `REPETICIONES = 2` en la primera celda de la sección 8.

### En local

Requiere Python 3.13 o superior y [uv](https://docs.astral.sh/uv/).

```bash
uv sync
uv run --with jupyter jupyter lab
```

También se puede abrir `relevancia_memorias_locomo.ipynb` en VS Code y elegir el kernel de `.venv`.

Para recalcular los embeddings desde cero (opcional), instalar [Ollama](https://ollama.com), ejecutar `ollama pull qwen3-embedding:0.6b` y borrar `data/embeddings.npz`.

## Archivos

| Archivo | Descripción |
|---|---|
| `relevancia_memorias_locomo.ipynb` | Notebook con todo el proyecto |
| `data/locomo10.json` | Dataset LoCoMo original |
| `data/embeddings.npz` | Embeddings de los 5.882 turnos y las 500 preguntas |
| `data/pares.csv` | Tabla de 10.000 pares con sus features y la etiqueta |
| `pyproject.toml`, `uv.lock` | Dependencias del entorno |

## Referencias

Las referencias completas, en formato IEEE, están al final del notebook.
