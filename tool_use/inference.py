"""Multi-turn inference loop with tool-use support for Phase 2."""

from typing import Any

import torch
from transformers import AutoTokenizer, PreTrainedModel

from rlm.inference import load_rlm_model
from tool_use.system_prompt import TOOL_USE_SYSTEM_PROMPT
from tool_use.tool_handler import execute_tool, parse_tool_call
from tool_use.tools import TOOL_SCHEMAS

MAX_TOOL_TURNS: int = 3
MAX_NEW_TOKENS: int = 1024


def generate_with_tools(
    prompt: str,
    model: PreTrainedModel,
    tokenizer: AutoTokenizer,
) -> dict[str, Any]:
    """Generate a response with tool-use support via a multi-turn loop.

    The model may call tools using ``<tool_call>`` tags. When detected, the tool
    is executed and the result is fed back as a ``role: tool`` message. The model
    then generates a final answer.

    Args:
        prompt: User question/prompt.
        model: The loaded LoRA-adapted model.
        tokenizer: The corresponding tokenizer.

    Returns:
        Dict with 'response' (final model text) and 'trace' (list of turn dicts).
    """
    messages: list[dict[str, str]] = [
        {"role": "system", "content": TOOL_USE_SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    trace: list[dict[str, Any]] = []
    device = next(model.parameters()).device
    generated_text = ""

    for _turn in range(MAX_TOOL_TURNS + 1):
        # Build input using Qwen's chat template with tool schemas injected
        input_text = tokenizer.apply_chat_template(
            messages,
            tools=TOOL_SCHEMAS,
            add_generation_prompt=True,
            tokenize=False,
        )

        inputs = tokenizer(input_text, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        input_len = inputs["input_ids"].shape[-1]

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=True,
                temperature=0.3,
                pad_token_id=tokenizer.eos_token_id,
            )

        generated_text = tokenizer.decode(
            output_ids[0][input_len:], skip_special_tokens=True
        )
        trace.append({"role": "assistant", "content": generated_text})

        # Check for tool call in the generated text
        parsed = parse_tool_call(generated_text)
        if parsed is None:
            # No tool call — this is the final answer
            return {"response": generated_text, "trace": trace}

        # Execute the tool and record in trace
        tool_name, tool_args = parsed
        tool_result = execute_tool(tool_name, tool_args)
        trace.append({
            "role": "tool",
            "tool_name": tool_name,
            "tool_args": tool_args,
            "content": tool_result,
        })

        # Append to conversation history for next generation turn
        messages.append({"role": "assistant", "content": generated_text})
        messages.append({"role": "tool", "content": tool_result})

    # Exhausted all turns without a direct final answer
    return {"response": generated_text, "trace": trace}


if __name__ == "__main__":
    m, tok = load_rlm_model()

    test_prompts = [
        "Calculate the creatinine clearance for a 65-year-old male weighing 80 kg "
        "with serum creatinine of 1.5 mg/dL.",
        "What are the warnings for ibuprofen?",
    ]
    for p in test_prompts:
        print("=" * 60)
        print(f"Prompt: {p}")
        result = generate_with_tools(p, m, tok)
        print(f"Response: {result['response']}")
        for step in result["trace"]:
            print(f"  [{step['role']}] {step.get('content', '')[:200]}")
        print()
