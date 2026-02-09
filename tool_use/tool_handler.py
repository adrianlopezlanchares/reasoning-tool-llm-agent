import re
import json
from tools import calculator, simulated_search

def parse_and_execute_tool_call(model_output):
    """
    Intenta detectar y ejecutar una llamada a herramienta en el output del modelo.
    
    Retorna:
    - El resultado de la herramienta (str) si hubo una llamada exitosa.
    - None si no se detectó ninguna llamada válida.
    """
    """
    <tool>
    name: calculator, input: "25 * 4 + 100"
    </tool>
    """
    # regex to get all text between <tool> and </tool>
    tool_calls = re.findall(r"<tool>(.*?)</tool>", model_output, re.DOTALL)

    for call in tool_calls:
        call = json.loads(call.strip())
        tool_name = call.get("name")
        tool_input = call.get("input")

        if tool_name == "calculator":
            return calculator(tool_input)
        elif tool_name == "simulated_search":
            return simulated_search(tool_input)
        else:
            print(f"Tool desconocida: {tool_name}")
            return f"Error: Tool desconocida '{tool_name}'"