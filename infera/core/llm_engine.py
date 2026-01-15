import atexit
from time import perf_counter
from typing import Dict, List, Union

import torch.multiprocessing as mp
from tqdm.auto import tqdm
from transformers import AutoTokenizer

from infera.config import InferaConfig
from infera.sampling import SamplingParams
from infera.core.sequence import Sequence
from infera.core.scheduler import Scheduler
from infera.core.model_runner import ModelRunner


class InferaEngine:
    """
    Core inference engine of Infera.

    Responsible for orchestrating scheduling, model execution,
    and request lifecycle management.
    """

    def __init__(self, config: InferaConfig):
        self.config = config

        # -----------------------------
        # Tokenizer
        # -----------------------------
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.model_path,
            use_fast=True,
        )

        # -----------------------------
        # Runtime & Scheduler
        # -----------------------------
        self.scheduler = Scheduler(config)

        # -----------------------------
        # Model Runners (TP)
        # -----------------------------
        self._init_model_runners()

        atexit.register(self.shutdown)

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def _init_model_runners(self) -> None:
        ctx = mp.get_context("spawn")
        self._workers = []
        self._events = []

        for rank in range(1, self.config.tensor_parallel_size):
            event = ctx.Event()
            proc = ctx.Process(
                target=ModelRunner,
                args=(self.config, rank, event),
            )
            proc.start()
            self._workers.append(proc)
            self._events.append(event)

        # Rank 0 runs in main process
        self.model_runner = ModelRunner(
            self.config,
            rank=0,
            peer_events=self._events,
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def shutdown(self) -> None:
        if hasattr(self, "model_runner"):
            self.model_runner.call("exit")
            del self.model_runner

        for p in getattr(self, "_workers", []):
            p.join()

    # ------------------------------------------------------------------
    # Request API
    # ------------------------------------------------------------------

    def add_request(
        self,
        prompt: Union[str, List[int]],
        sampling_params: SamplingParams,
    ) -> None:
        if isinstance(prompt, str):
            prompt = self.tokenizer.encode(prompt)

        seq = Sequence(
            prompt_token_ids=prompt,
            sampling_params=sampling_params,
        )
        self.scheduler.add(seq)

    # ------------------------------------------------------------------
    # Engine Step
    # ------------------------------------------------------------------

    def step(self) -> Dict[str, int]:
        """
        Execute one scheduling step.

        Returns:
            dict with runtime statistics:
              - num_prefill_tokens
              - num_decode_tokens
        """
        seqs, is_prefill = self.scheduler.schedule()

        if not seqs:
            return {}

        token_ids = self.model_runner.call(
            "run",
            seqs,
            is_prefill,
        )

        self.scheduler.postprocess(seqs, token_ids)

        if is_prefill:
            return {"num_prefill_tokens": sum(len(s) for s in seqs)}
        else:
            return {"num_decode_tokens": len(seqs)}

    def is_finished(self) -> bool:
        return self.scheduler.is_finished()

    # ------------------------------------------------------------------
    # High-level Generation API
    # ------------------------------------------------------------------

    def generate(
        self,
        prompts: List[Union[str, List[int]]],
        sampling_params: Union[SamplingParams, List[SamplingParams]],
        show_progress: bool = True,
    ) -> List[Dict]:
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)

        for prompt, sp in zip(prompts, sampling_params):
            self.add_request(prompt, sp)

        outputs: Dict[int, List[int]] = {}

        pbar = None
        if show_progress:
            pbar = tqdm(total=len(prompts), desc="Infera Generating", dynamic_ncols=True)

        while not self.is_finished():
            t0 = perf_counter()
            stats = self.step()
            dt = perf_counter() - t0

            if pbar and stats:
                postfix = {}
                if "num_prefill_tokens" in stats:
                    postfix["Prefill"] = f"{int(stats['num_prefill_tokens'] / dt)} tok/s"
                if "num_decode_tokens" in stats:
                    postfix["Decode"] = f"{int(stats['num_decode_tokens'] / dt)} tok/s"
                pbar.set_postfix(postfix)

            for seq in self.scheduler.finished_sequences():
                outputs[seq.seq_id] = seq.completion_token_ids
                if pbar:
                    pbar.update(1)

        if pbar:
            pbar.close()

        return [
            {
                "text": self.tokenizer.decode(token_ids),
                "token_ids": token_ids,
            }
            for _, token_ids in sorted(outputs.items())
        ]
