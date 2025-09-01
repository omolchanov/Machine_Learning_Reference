import os

import warnings
warnings.filterwarnings('ignore')

from together import Together


TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")
API_URL = "https://api.together.xyz/v1"
MODEL = "deepseek-ai/DeepSeek-R1-Distill-Llama-70B-free"

client = Together()


# Universal prompt template
def create_prompt(prompt_type="zero-shot", role="", task="", format="", examples="", instructions=""):
    """
    Creates prompts for different prompting strategies
    """
    if prompt_type.lower() == "zero-shot":
        return f"""
        You are {role}.
        Task: {task}
        Format: {format}
        Instructions: {instructions}
        """.strip()

    elif prompt_type.lower() == "few-shot":
        return f"""
        You are {role}.
        Task: {task}
        Format: {format}
        Here are some examples:
        {examples}

        Now continue with more cases.
        """.strip()

    elif prompt_type.lower() == "role":
        return f"""
        Act as {role}.
        Your job: {task}
        Please respond in {format}.
        Extra rules: {instructions}
        """.strip()

    elif prompt_type.lower() == "chain-of-thought":
        return f"""
        You are {role}.
        Task: {task}
        Please reason step by step before giving the final answer.
        Format: {format}
        Instructions: {instructions}
        """.strip()

    elif prompt_type.lower() == "react":
        return f"""
        You are {role}.
        Task: {task}
        Use ReAct: think (reasoning) → act (output) steps.
        Format: {format}
        Instructions: {instructions}
        """.strip()

    else:
        return f"You are {role}. Task: {task}. Format: {format}"


# Call Together API
def ask_together(prompt: str):
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=500,
        temperature=0.7
    )
    return response.choices[0].message.content


if __name__ == "__main__":
    # Try Few-shot
    prompt = create_prompt(
        prompt_type="few-shot",
        role="",
        task="Write 5 test cases for function is_even(n)",
        format="JSON",
        examples='[{"input":2, "expected":true}, {"input":3, "expected":false}]',
        instructions="Add both positive and negative scenarios"
    )

    result = ask_together(prompt)
    print("\n🔹 Generated Tests (Few-shot):\n")
    print(result)

    # Try Chain-of-thought
    prompt = create_prompt(
        prompt_type="chain-of-thought",
        role="QA engineer",
        task="Write test cases for is_even(n).",
        format="JSON",
        instructions="Show your reasoning before giving final JSON."
    )

    result = ask_together(prompt)
    print("\n🔹 Generated Tests (CoT):\n")
    print(result)

    # Try react
    prompt = create_prompt(
        prompt_type="react",
        task="Write test cases for is_even(n).",
        format="table"
    )

    result = ask_together(prompt)
    print("\n🔹 Generated Tests (React):\n")
    print(result)