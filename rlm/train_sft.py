import os
import subprocess

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig, TaskType
from trl import SFTTrainer

# Configuration
MODEL_NAME: str = "Qwen/Qwen2.5-7B-Instruct"
DATASET_NAME: str = "gsm8k"
OUTPUT_DIR: str = os.environ.get("SFT_MODEL_PATH", "./weights/sft_lora")

# HYPERPARAMETERS
EPOCHS: int = 3
BATCH_SIZE: int = 8
LR: float = 5e-6
LORA_RANK: int = 8
LORA_ALPHA: int = 16

def get_freest_gpu():
    try:        
        # Run nvidia-smi to get memory usage
        result = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.free,index", "--format=csv,nounits,noheader"],encoding="utf-8")        
        # Parse output: "12345, 0" -> (12345 MB, GPU 0)        
        gpu_memory = []
        for line in result.strip().split('\n'): 
            free_mem, index = line.split(',') 
            gpu_memory.append((int(free_mem), int(index)))
        # Sort by free memory (descending)        
        gpu_memory.sort(key=lambda x: x[0], reverse=True)        
        best_gpu_index = gpu_memory[0][1] 
        best_gpu_mem = gpu_memory[0][0] 
        print(f"✅ Auto-selected GPU {best_gpu_index} with {best_gpu_mem}MB free.") 
        return str(best_gpu_index) 
    except Exception as e: 
        print(f"⚠️ Could not detect GPUs automatically: {e}") 
        return "0" # Fallback
    

os.environ["CUDA_VISIBLE_DEVICES"] = get_freest_gpu()
print(f"Using GPU: {os.environ['CUDA_VISIBLE_DEVICES']}")

def formatting_prompts_func(example: dict) -> str:
    question = example["question"]
    answer_full = example["answer"]

    if "####" in answer_full:
        reasoning, final_answer = answer_full.split("####", 1)
        reasoning = reasoning.strip()
        final_answer = final_answer.strip()
    else:
        reasoning = answer_full.strip()
        final_answer = ""

    text = (
        f"User: {question}\n"
        f"Assistant: <think>\n{reasoning}\n</think>\n"
        f"Response: {final_answer}"
    )
    return text


def train():
    # 1. Load Model and Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, device_map={"": 0}, dtype=torch.bfloat16)

    # 2. Configure LoRA
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.01,
        bias="none",
    )

    # 3. Load Dataset
    dataset = load_dataset(DATASET_NAME, name="main", split="train")

    # 4. Configure Training
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=2,
        learning_rate=LR,
        fp16=True,
        logging_steps=100,
    )

    # 5. Initialize SFTTrainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=peft_config,
        args=training_args,
        formatting_func=formatting_prompts_func,
        processing_class=tokenizer,
    )

    # 6. Train and save
    trainer.train()
    trainer.save_model(OUTPUT_DIR)
    print("SFT Training finished")

if __name__ == "__main__":
    train()