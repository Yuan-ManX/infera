import os
from typing import List

from transformers import AutoTokenizer

from infera.llm import LLM
from infera.sampling import SamplingParams


# ======================
# Configuration
# ======================
MODEL_PATH = os.path.expanduser("~/huggingface/Qwen3/")
TEMPERATURE = 0.6
MAX_TOKENS = 256


def build_chat_prompts(
    tokenizer: AutoTokenizer,
    user_prompts: List[str],
) -> List[str]:
    """
    Convert plain user prompts into model-specific chat format.
    """
    return [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for prompt in user_prompts
    ]


def main():
    # ----------------------
    # Initialize tokenizer
    # ----------------------
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

    # ----------------------
    # Initialize LLM
    # ----------------------
    llm = LLM(
        MODEL_PATH,
        enforce_eager=True,
        tensor_parallel_size=1,
    )

    # ----------------------
    # Sampling parameters
    # ----------------------
    sampling_params = SamplingParams(
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
    )

    # ----------------------
    # Prompts
    # ----------------------
    user_prompts = [
        "1 + 1 =",
        "who are you",
    ]

    prompts = build_chat_prompts(tokenizer, user_prompts)

    # ----------------------
    # Inference
    # ----------------------
    outputs = llm.generate(prompts, sampling_params)

    # ----------------------
    # Print results
    # ----------------------
    for user_prompt, formatted_prompt, output in zip(
        user_prompts, prompts, outputs
    ):
        print("\n" + "=" * 40)
        print(f"User Prompt: {user_prompt}")
        print("-" * 40)
        print(f"Formatted Prompt:\n{formatted_prompt}")
        print("-" * 40)
        print(f"Completion:\n{output.get('text', '')}")


if __name__ == "__main__":
    main()

