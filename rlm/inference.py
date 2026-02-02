import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Ruta a tu modelo final de fase 1
MODEL_PATH = os.environ.get("FINAL_MODEL_PATH", "./weights/final_rlm_lora")
BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"

def load_rlm_model():
    # TODO: Cargar el modelo base y el adaptador LoRA
    print(f"Cargando modelo RLM desde {MODEL_PATH}...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=torch.float16, device_map="auto")
    model = PeftModel.from_pretrained(base_model, MODEL_PATH)
    return model, tokenizer

def generate_reasoning(prompt, model, tokenizer):
    """
    Genera una respuesta que incluye el razonamiento (CoT).
    """
    # TODO: Implementar la generación
    input_ids = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**input_ids, max_new_tokens=256, eos_token_id=tokenizer.eos_token_id)
    input_len = input_ids["input_ids"].shape[1]
    generated = outputs[0][input_len:]

    return tokenizer.decode(generated, skip_special_tokens=True)

if __name__ == "__main__":
    # Prueba local
    model, tokenizer = load_rlm_model()
    test_prompt = "Si tengo 3 manzanas y me dan el doble de las que tengo menos una, ¿cuántas tengo?"
    print(generate_reasoning(test_prompt, model, tokenizer))
