import os
import warnings

os.environ["HF_HOME"] = "/root/.cache/huggingface"
warnings.filterwarnings(
    "ignore", category=FutureWarning, module="transformers.utils.hub"
)

import torch  # noqa: E402
from datasets import load_dataset  # noqa: E402
from peft import LoraConfig  # noqa: E402
from transformers import (  # noqa: E402
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import SFTConfig, SFTTrainer  # noqa: E402

# TODO: Configuración del modelo y dataset
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"  # O un modelo más pequeño si es necesario
DATASET_NAME = "gsm8k"
OUTPUT_DIR = "./weights/sft_lora"


def formatting_prompts_func(examples):
    output_texts = []
    # Iterate over the batch
    for question, answer in zip(examples["question"], examples["answer"]):
        # Extract reasoning if available
        if "####" in answer:
            parts = answer.split("####")
            reasoning = parts[0].strip()
            final_answer = parts[1].strip()
        else:
            reasoning = answer
            final_answer = "Error parsing"

        # Construct the prompt
        text = (
            f"<|im_start|>user\n{question}<|im_end|>\n"
            f"<|im_start|>assistant\n<think>\n{reasoning}\n</think>\n"
            f"La respuesta es {final_answer}<|im_end|>"
        )
        output_texts.append(text)

    return {"text": output_texts}


def train():
    # 1. Cargar Modelo y Tokenizer (usar cuantización 4bit/8bit si es necesario)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16,
    )

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
    dataset = load_dataset(DATASET_NAME, "main", split="train")

    print("Formatting dataset...")
    dataset = dataset.map(formatting_prompts_func, batched=True)

    sft_config = SFTConfig(
        output_dir=OUTPUT_DIR,
        # dataset_text_field="text",
        # max_seq_length=1024,
        num_train_epochs=1,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        fp16=True,
        logging_steps=10,
        save_steps=100,
        save_total_limit=2,
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=peft_config,
        processing_class=tokenizer,  # Use 'processing_class' for trl >= 0.13.0
        args=sft_config,  # Pass the SFTConfig object
    )

    # 6. Entrenar y guardar
    trainer.train()
    trainer.save_model(OUTPUT_DIR)
    print("SFT Training finished (TODO: Implement)")


if __name__ == "__main__":
    train()
