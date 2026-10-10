import logging
import re
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from . import prompts
from .vlm import parse_action

logger = logging.getLogger(__name__)

_ACTIONS = ("delete", "stop", "continue")


def _vote(actions: Sequence[str]) -> Tuple[str, Dict[str, float]]:
    votes = {a: 0.0 for a in _ACTIONS}
    for a in actions:
        if a in votes:
            votes[a] += 1.0
    total = sum(votes.values())
    if total == 0:
        return "continue", {"continue": 1.0, "stop": 0.0, "delete": 0.0}
    votes = {a: v / total for a, v in votes.items()}
    return max(votes, key=votes.get), votes


class MLLMJudge:


    def __init__(self, backend):
        self.backend = backend

    def judge(self, question, caption, position, root_caption="", coverage=""):
        return self.backend.judge(question, caption, position, root_caption, coverage)


class LoRAJudge:


    def __init__(self, cfg):
        self.cfg = cfg
        dtype = getattr(torch, cfg.dtype)
        logger.info("Loading judge LLM %s", cfg.judge_llm_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            cfg.judge_llm_path, dtype=dtype, attn_implementation=cfg.attn_impl
        )
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.judge_llm_path)
        if not cfg.judge_lora_path:
            raise ValueError("The policy adapter checkpoint is required")
        merge_lora_(self.model, cfg.judge_lora_path)
        self.model.eval().to(cfg.judge_device or cfg.device)

    @torch.no_grad()
    def judge(self, question, caption, position, root_caption="", coverage=""):
        cfg = self.cfg
        if cfg.judge_prompt_style == "action_only":
            prompt = prompts.JUDGE_PROMPT_ACTION_ONLY.format(
                question=question, caption=caption, position=position
            )
        else:
            prompt = prompts.JUDGE_PROMPT.format(
                question=question, caption=caption, position=position,
                max_depth=cfg.max_depth, root_caption=root_caption or "(not available)",
                coverage=coverage or "(unknown)",
            )
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        k = max(1, cfg.judge_samples)
        out = self.model.generate(
            **inputs,
            max_new_tokens=cfg.judge_max_new_tokens,
            do_sample=True,
            temperature=cfg.judge_temperature,
            top_k=50,
            top_p=cfg.judge_top_p,
            num_return_sequences=k,
            repetition_penalty=1.1,
            pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
        )
        texts = self.tokenizer.batch_decode(
            out[:, inputs.input_ids.shape[1] :], skip_special_tokens=True
        )

        actions = []
        for t in texts:
            a = parse_action(t)
            if cfg.judge_prompt_style == "action_only" and a not in ("stop", "delete"):

                a = "continue"
            actions.append(a)
        action, votes = _vote(actions)
        best = next((t for t, a in zip(texts, actions) if a == action), texts[0])
        return action, best.strip(), votes


def merge_lora_(model, lora_dir: str, sign: float = 1.0) -> int:


    import json
    import os

    from safetensors.torch import load_file

    with open(os.path.join(lora_dir, "adapter_config.json")) as f:
        acfg = json.load(f)
    unsupported = [k for k in ("use_dora", "fan_in_fan_out", "rank_pattern", "alpha_pattern", "modules_to_save", "lora_bias") if acfg.get(k)]
    if unsupported or acfg.get("bias", "none") != "none":
        raise ValueError(f"Adapter needs PEFT-specific merging: {unsupported or ['bias']}")
    r, alpha = int(acfg["r"]), float(acfg["lora_alpha"])
    if acfg.get("use_rslora"):
        scale = alpha / (r ** 0.5)
    else:
        scale = alpha / r
    scale *= sign

    state = load_file(os.path.join(lora_dir, "adapter_model.safetensors"))
    pairs: Dict[str, Dict[str, torch.Tensor]] = {}
    for key, tensor in state.items():
        if ".lora_A." in key:
            name, side = key.split(".lora_A.")[0], "A"
        elif ".lora_B." in key:
            name, side = key.split(".lora_B.")[0], "B"
        else:
            raise ValueError(f"Unsupported adapter tensor: {key}")
        name = re.sub(r"^base_model\.model\.", "", name)
        pairs.setdefault(name, {})[side] = tensor


    for name, ab in pairs.items():
        if set(ab) != {"A", "B"}:
            raise ValueError(f"Incomplete LoRA pair: {name}")
        module = model.get_submodule(name)
        a, b = ab["A"], ab["B"]
        if a.ndim != 2 or b.ndim != 2 or a.shape[0] != r or b.shape[1] != r or (b.shape[0], a.shape[1]) != tuple(module.weight.shape):
            raise ValueError(f"LoRA shape mismatch: {name}")
    merged = 0
    for name, ab in pairs.items():
        if "A" not in ab or "B" not in ab:
            logger.warning("incomplete LoRA pair for %s, skipped", name)
            continue
        try:
            module = model.get_submodule(name)
        except AttributeError:
            logger.warning("no submodule %s in base model, skipped", name)
            continue
        w = module.weight
        delta = (ab["B"].float() @ ab["A"].float()) * scale
        if delta.shape != tuple(w.shape):
            logger.warning("shape mismatch %s: %s vs %s", name, delta.shape, tuple(w.shape))
            continue
        w.data.add_(delta.to(w.dtype).to(w.device))
        merged += 1

    logger.info("merged %d LoRA modules from %s (scale=%.3f)", merged, lora_dir, scale)
    if merged == 0:
        raise RuntimeError(f"no LoRA module merged from {lora_dir}")
    return merged


def unmerge_lora_(model, lora_dir: str) -> int:
    return merge_lora_(model, lora_dir, sign=-1.0)


def build_judge(cfg, backend):
    if cfg.judge_backend == "lora_llm":
        return LoRAJudge(cfg)
    return MLLMJudge(backend)
