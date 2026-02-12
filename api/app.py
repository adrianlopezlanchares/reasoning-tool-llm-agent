import os
import subprocess
import sys

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Añadir el directorio raíz al path para poder importar los módulos de las fases
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# --- IMPORTACIONES DE LOS MÓDULOS DE LOS ALUMNOS ---
# TODO: Descomentar a medida que se implementen las fases
from rlm.inference import generate_reasoning, load_rlm_model
from tool_use.inference import generate_with_tools
from tool_use.tool_handler import parse_and_execute_tool_call


def get_freest_gpu():
    try:
        # Run nvidia-smi to get memory usage
        result = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=memory.free,index",
                "--format=csv,nounits,noheader",
            ],
            encoding="utf-8",
        )
        # Parse output: "12345, 0" -> (12345 MB, GPU 0)
        gpu_memory = []
        for line in result.strip().split("\n"):
            free_mem, index = line.split(",")
            gpu_memory.append((int(free_mem), int(index)))
        # Sort by free memory (descending)
        gpu_memory.sort(key=lambda x: x[0], reverse=True)
        best_gpu_index = gpu_memory[0][1]
        best_gpu_mem = gpu_memory[0][0]
        print(f"✅ Auto-selected GPU {best_gpu_index} with {best_gpu_mem}MB free.")
        return str(best_gpu_index)
    except Exception as e:
        print(f"⚠️ Could not detect GPUs automatically: {e}")
        return "0"  # Fallback


os.environ["CUDA_VISIBLE_DEVICES"] = get_freest_gpu()
print(f"Using GPU: {os.environ['CUDA_VISIBLE_DEVICES']}")

# from rag.rag_engine import retrieve_context, format_rag_prompt
# from react.agent import ReActAgent

app = FastAPI(
    title="Práctica Master: Modelos Generativos Profundos",
    description="API para evaluar las 4 fases de la práctica.",
)

# --- Variables Globales (Modelos) ---
# Se cargan al inicio para no recargarlos en cada petición
MODEL = None
TOKENIZER = None
AGENT = None


@app.on_event("startup")
async def startup_event():
    global MODEL, TOKENIZER, AGENT
    print("Inicializando API...")

    # Cargar el modelo de la Fase 1 (RLM)
    try:
        MODEL, TOKENIZER = load_rlm_model()
        print("Modelo RLM de Fase 1 cargado correctamente.")
    except Exception as e:
        print(f"Error al cargar modelo RLM: {e}")
        MODEL, TOKENIZER = None, None

    # TODO: Descomentar cuando se implemente ReAct
    # if MODEL:
    #     AGENT = ReActAgent(MODEL, TOKENIZER)

    print("Inicialización de API completada.")


# --- Modelos de Pydantic para Request/Response ---
class QueryRequest(BaseModel):
    prompt: str


class GenericResponse(BaseModel):
    response: str
    trace: list[dict] = []
    details: dict = {}


# ================= ENDPOINTS DE EVALUACIÓN =================


# --- FASE 1: Razonamiento (RLM) ---
@app.post("/phase1/reasoning", response_model=GenericResponse, tags=["Fase 1"])
async def phase1_endpoint(request: QueryRequest):
    """
    Evalúa el modelo RLM. Debe devolver la respuesta con el razonamiento (CoT) visible.
    """
    if not MODEL or not TOKENIZER:
        return {
            "response": "ERROR: Modelo de Fase 1 no cargado.",
            "details": {"status": "model_not_loaded"},
        }

    try:
        # Usar la función de inferencia de Fase 1
        response_text = generate_reasoning(request.prompt, MODEL, TOKENIZER)
        return {
            "response": response_text,
            "trace": [{"step": 0, "content": response_text}],
            "details": {"stage": "sft_grpo", "status": "success"},
        }
    except Exception as e:
        return {
            "response": f"ERROR durante la generación: {str(e)}",
            "trace": [],
            "details": {"stage": "sft_grpo", "status": "error", "error": str(e)},
        }


# --- FASE 2: Tool Use ---
@app.post("/phase2/tools", response_model=GenericResponse, tags=["Fase 2"])
async def phase2_endpoint(request: QueryRequest):
    """
    Evalúa la capacidad de llamar herramientas.
    Si el prompt requiere una herramienta, debe devolver la ejecución simulada.
    """
    if not MODEL or not TOKENIZER:
        return {
            "response": "ERROR: Modelo de Fase 2 no cargado.",
            "details": {"status": "model_not_loaded"},
        }

    try:
        # Use the multi-turn tool-use inference loop
        result = generate_with_tools(request.prompt, MODEL, TOKENIZER)

        print(f"Tool-use response: {result['response']}")
        print(f"Trace: {result['trace']}")

        # Check if any tools were called by looking at the trace
        tool_called = any(step.get("role") == "tool" for step in result["trace"])

        return {
            "response": result["response"],
            "trace": result["trace"],
            "details": {"tool_called": tool_called, "status": "success"},
        }
    except Exception as e:
        return {
            "response": f"ERROR during tool-use generation: {str(e)}",
            "trace": [],
            "details": {"status": "error", "error": str(e)},
        }


# --- FASE 3: RAG ---
@app.post("/phase3/rag", response_model=GenericResponse, tags=["Fase 3"])
async def phase3_endpoint(request: QueryRequest):
    """
    Evalúa el RAG. Debe recuperar contexto de los documentos y responder.
    """
    # TODO: Implementar lógica RAG
    # 1. Recuperar contexto
    # context_list = retrieve_context(request.prompt)
    # 2. Formatear prompt
    # rag_prompt = format_rag_prompt(request.prompt, context_list)
    # 3. Generar con el modelo (opcional, o devolver solo el contexto recuperado para evaluar)

    return {
        "response": "Placeholder Fase 3 (RAG)",
        "details": {"retrieved_docs": ["doc1_placeholder", "doc2_placeholder"]},
    }


# --- FASE 4: Agente ReAct ---
@app.post("/phase4/agent", tags=["Fase 4"])
async def phase4_endpoint(request: QueryRequest):
    """
    Evalúa el agente completo. Devuelve la respuesta final y la traza de ejecución.
    """
    if not AGENT:
        return {"final_answer": "ERROR: Agente no inicializado.", "trace": []}

    # TODO: Ejecutar agente
    # result = AGENT.run(request.prompt)
    result = {
        "final_answer": "Placeholder Fase 4 Agent",
        "trace": [{"step": 0, "content": "..."}],
    }  # TODO remove

    return result


if __name__ == "__main__":
    # Para correr localmente: python api/app.py
    uvicorn.run(app, host="0.0.0.0", port=8000)
