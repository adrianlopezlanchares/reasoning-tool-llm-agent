# --- Implementación de las herramientas ---

import os

import numexpr
from dotenv import load_dotenv
from langchain.tools import tool

from rlm.inference import load_rlm_model, generate_reasoning

load_dotenv()

MODEL_PATH = os.environ.get("FINAL_MODEL_PATH", "./weights/final_rlm_lora")
BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"

@tool
def calculator(expression: str) -> str:
    """Evalúa una expresión matemática simple. Útil para realizar cálculos aritméticos."""
    try:
        return str(numexpr.evaluate(expression))
    except Exception as e:
        return f"Error calculando: {e}"


@tool
def simulated_search(query: str) -> str:
    """Busca información en una base de datos. SIEMPRE usa esta herramienta para buscar información sobre personas, lugares, tecnología o cualquier dato factual. Input: la consulta de búsqueda."""
    query_lower = query.lower()
    if "hermano" in query_lower and "miguel" in query_lower:
        return "Miguel tiene un hermano llamado Juan."
    elif "capital" in query_lower and "francia" in query_lower:
        return "La capital de Francia es París."
    elif "python" in query_lower:
        return "Python es un lenguaje de programación de alto nivel."
    else:
        return "No se encontraron resultados relevantes en el buscador simulado."


# Lista de herramientas disponibles
tools = [calculator, simulated_search]


SYSTEM_PROMPT = """Eres un asistente que SIEMPRE usa las herramientas disponibles para responder preguntas.

REGLAS IMPORTANTES:
1. Para CUALQUIER pregunta sobre personas, lugares, datos o hechos, USA la herramienta simulated_search PRIMERO.
2. Para cálculos matemáticos, USA la herramienta calculator.
3. NUNCA respondas basándote en tu conocimiento propio sin antes consultar las herramientas.
4. Si una herramienta no devuelve resultados, entonces puedes indicar que no encontraste la información."""


def main():
    """Ejemplo de uso de un agente con herramientas usando LangChain y Azure OpenAI."""
    
    model, tokenizer = load_rlm_model()
    
    # # Ejemplo 1: Pregunta que requiere cálculo
    print("=" * 60)
    print("Ejemplo 1: Cálculo matemático")
    print("=" * 60)
    result = generate_reasoning("¿Cuánto es 25 * 4 + 100?", model, tokenizer)
    print(f"Respuesta: {result['messages'][-1].content}\n")
    
    # Ejemplo 2: Pregunta que requiere búsqueda
    print("=" * 60)
    print("Ejemplo 2: Búsqueda de información")
    print("=" * 60)
    result = generate_reasoning("¿Quién es el hermano de Miguel?", model, tokenizer)
    print(f"Respuesta: {result['messages'][-1].content}\n")
    


if __name__ == "__main__":
    main()
