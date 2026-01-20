import re

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from datasets import load_dataset

from typing import List

# Configuración
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
SFT_ADAPTER_PATH = "./weights/sft_lora"  # Ruta al modelo de la Fase 1, Parte 1
OUTPUT_DIR = "./weights/final_rlm_lora"
GRPO_GROUP_SIZE = 4  # N respuestas por pregunta
DATASET_NAME = "gsm8k"

# pre-compile re matcher
answer_regex = re.compile(r'La respuesta es ([\d\.]+)')


def reward_function(generated_text: str, ground_truth_answer) -> float:
    """Extrae una respuesta numérica simple y compara con la respuesta real.

    Devuelve 1.0 si es exactamente igual, 0.0 en otro caso. Se puede mejorar.
    """
    match = answer_regex.search(generated_text)
    if match:
        generated_answer = match.group(1)
        if generated_answer == str(ground_truth_answer):
            return 1.0
    return 0.0


def _build_prompt(question: str) -> str:
    return f"User: {question}\nAssistant: "


def train_grpo(
    epochs: int = 1,
    batch_size: int = 2,
    group_size: int = GRPO_GROUP_SIZE,
    lr: float = 1e-5,
    max_new_tokens: int = 64,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Cargar modelo base y aplicarle el adaptador SFT
    base_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16)
    model = PeftModel.from_pretrained(base_model, SFT_ADAPTER_PATH, is_trainable=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    tokenizer.pad_token = tokenizer.eos_token

    model = model.to(device)
    model.train()

    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

    # 2. Cargar dataset y preparar iteración simple
    dataset = load_dataset(DATASET_NAME, split="train")
    examples = [
        {"question": ex["question"], "answer": ex.get("answer", "")} for ex in dataset
    ]

    n_examples = len(examples)

    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs} — examples {n_examples}")

        for start in range(0, n_examples, batch_size):
            batch = examples[start : start + batch_size]

            # For each question in batch generate `group_size` responses
            all_generated_ids = []
            all_generated_texts = []
            all_ground_truths = []

            for ex in batch:
                prompt = _build_prompt(ex["question"])
                input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

                with torch.no_grad():
                    gen_ids = model.generate(
                        input_ids=input_ids,
                        do_sample=True,
                        temperature=1.0,
                        top_p=0.95,
                        max_new_tokens=max_new_tokens,
                        num_return_sequences=group_size,
                        pad_token_id=tokenizer.eos_token_id,
                    )

                # model.generate returns sequences with prompt + continuation; keep full ids
                for g in gen_ids:
                    all_generated_ids.append(g.cpu())
                    text = tokenizer.decode(g, skip_special_tokens=True)
                    all_generated_texts.append(text)
                    all_ground_truths.append(ex["answer"])

            # 3. Compute rewards per generated sequence, grouped per question
            rewards = []
            for text, gt in zip(all_generated_texts, all_ground_truths):
                r = reward_function(text, gt)
                rewards.append(r)

            rewards = torch.tensor(rewards, dtype=torch.float32, device=device)

            # 4. Compute log-probabilities under current model for each generated sequence
            # We'll compute per-sequence log_prob by summing token log-probs for the generated continuation
            log_probs = []
            for gen_ids in all_generated_ids:
                # gen_ids is on cpu
                gen_ids = gen_ids.to(device)
                # prepare inputs and targets (predict next token)
                input_ids = gen_ids[:-1].unsqueeze(0)
                target_ids = gen_ids[1:].unsqueeze(0)

                outputs = model.base_model(
                    input_ids=input_ids,
                    return_dict=True,
                )
                logits = outputs.logits  # (1, seq_len, vocab)
                logprobs = F.log_softmax(logits, dim=-1)

                # gather logprob of each target token
                tgt_logprobs = logprobs.gather(2, target_ids.unsqueeze(-1)).squeeze(-1)
                # sum over tokens -> sequence log_prob
                seq_logprob = tgt_logprobs.sum()
                log_probs.append(seq_logprob)

            log_probs = torch.stack(log_probs)  # (batch_size * group_size,)

            # 5. Compute advantages per group (normalize within each question group)
            advantages = []
            idx = 0
            for _ in batch:
                group_rewards = rewards[idx : idx + group_size]
                mean_r = group_rewards.mean()
                std_r = group_rewards.std()
                if std_r == 0:
                    adv = group_rewards - mean_r
                else:
                    adv = (group_rewards - mean_r) / (std_r + 1e-8)
                advantages.append(adv)
                idx += group_size

            advantages = torch.cat(advantages).to(device)

            # 6. Policy gradient loss (REINFORCE-like): - log_prob * advantage
            loss = -(log_probs * advantages).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # end epoch
        # optionally save intermediate adapter
        model.save_pretrained(OUTPUT_DIR)
        tokenizer.save_pretrained(OUTPUT_DIR)

    print("GRPO Training finished — model saved to", OUTPUT_DIR)


if __name__ == "__main__":
    train_grpo()
