SYSTEM_PROMPT = """
    You are ahelpful assistant, and you should ALWAYS respond in the following format:

    Assistant: <think>
    {reasoning}
    </think>
    <answer>
    {final}
    </answer>

    Do not put anything outside <think> and <answer>.
"""