import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from rlm.system_prompt import SYSTEM_PROMPT

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
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt"
    )

    # inputs puede ser un tensor (input_ids) o un dict, según tokenizer/modelo
    if isinstance(inputs, torch.Tensor):
        inputs = {"input_ids": inputs}

    # Mover a GPU si aplica
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    input_len = inputs["input_ids"].shape[-1]

    output = model.generate(
        **inputs,
        max_new_tokens=512,
        do_sample=True,
        temperature=0.3,
        pad_token_id=tokenizer.eos_token_id,
    )

    return tokenizer.decode(output[0][input_len:], skip_special_tokens=True)

if __name__ == "__main__":
    # Prueba local
    model, tokenizer = load_rlm_model()
    test_prompt = "Si tengo 3 manzanas y me dan el doble de las que tengo menos una, ¿cuántas tengo?"
    print(generate_reasoning(test_prompt, model, tokenizer))
