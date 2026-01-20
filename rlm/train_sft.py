import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig
from trl import SFTTrainer

# TODO: Configuración del modelo y dataset
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct" # O un modelo más pequeño si es necesario
DATASET_NAME = "gsm8k"
OUTPUT_DIR = "./weights/sft_lora"

def formatting_prompts_func(example):
    output_texts = []
    # TODO: Implementar la lógica para formatear el dataset.
    text = f"User: {example['question']}\nAssistant: <think>{example['reasoning']}</think> La respuesta es {example['answer']}"
    output_texts.append(text)

    return output_texts

def train():
    # 1. Cargar Modelo y Tokenizer (usar cuantización 4bit/8bit si es necesario)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16)

    # 2. Configurar LoRA
    peft_config = LoraConfig(
        r=8,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj"],
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM",
        use_rslora=True,
    )

    # 3. Cargar Dataset
    dataset = load_dataset(DATASET_NAME, split="train")

    # 4. Configurar Entrenamiento
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=1,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=2,
        learning_rate=2e-4,
        fp16=True,
        logging_steps=10,
        save_steps=100,
        save_total_limit=2,
    )

    # 5. Inicializar SFTTrainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=peft_config,
        tokenizer=tokenizer,
        args=training_args,
        formatting_func=formatting_prompts_func,
    )

    # 6. Entrenar y guardar
    trainer.train()
    trainer.save_model(OUTPUT_DIR)
    print("SFT Training finished (TODO: Implement)")

if __name__ == "__main__":
    train()
