import os
import time
import random
import torch

from infera.llm import LLM
from infera.sampling import SamplingParams


# ======================
# Benchmark Configuration
# ======================
SEED = 0

NUM_SEQS = 256
MAX_INPUT_LEN = 1024
MAX_OUTPUT_LEN = 1024
MAX_MODEL_LEN = 4096

MODEL_PATH = os.path.expanduser("~/huggingface/Qwen3/")


def set_global_seed(seed: int):
    """Ensure reproducibility across Python & Torch."""
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_random_prompts(
    num_seqs: int,
    max_input_len: int,
    vocab_size: int = 10000,
):
    return [
        [random.randint(0, vocab_size) for _ in range(random.randint(100, max_input_len))]
        for _ in range(num_seqs)
    ]


def build_sampling_params(
    num_seqs: int,
    max_output_len: int,
):
    return [
        SamplingParams(
            temperature=0.6,
            ignore_eos=True,
            max_tokens=random.randint(100, max_output_len),
        )
        for _ in range(num_seqs)
    ]


def main():
    # ----------------------
    # Reproducibility
    # ----------------------
    set_global_seed(SEED)

    # ----------------------
    # Initialize LLM
    # ----------------------
    llm = LLM(
        MODEL_PATH,
        enforce_eager=False,
        max_model_len=MAX_MODEL_LEN,
    )

    # ----------------------
    # Prepare Inputs
    # ----------------------
    prompt_token_ids = build_random_prompts(
        num_seqs=NUM_SEQS,
        max_input_len=MAX_INPUT_LEN,
    )

    sampling_params = build_sampling_params(
        num_seqs=NUM_SEQS,
        max_output_len=MAX_OUTPUT_LEN,
    )

    # ----------------------
    # Warm-up (exclude timing)
    # ----------------------
    llm.generate(
        ["Benchmark warmup"],
        SamplingParams(max_tokens=1),
    )

    # ----------------------
    # Benchmark
    # ----------------------
    start_time = time.time()
    llm.generate(
        prompt_token_ids,
        sampling_params,
        use_tqdm=False,
    )
    elapsed = time.time() - start_time

    total_tokens = sum(sp.max_tokens for sp in sampling_params)
    throughput = total_tokens / elapsed

    print(
        f"Total Tokens: {total_tokens} | "
        f"Time: {elapsed:.2f}s | "
        f"Throughput: {throughput:.2f} tok/s"
    )


if __name__ == "__main__":
    main()
