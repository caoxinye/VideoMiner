import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoProcessor

from .vlm import QwenVLBackend


class VLLMBackend(QwenVLBackend):
    def close(self):

        self.engine.llm_engine.engine_core.shutdown()

    def __init__(self, cfg, *, gpu_memory_utilization=0.45, max_model_len=8192,
                 max_num_seqs=4, enforce_eager=True):
        from vllm import LLM

        if cfg.visual_input_mode != 'video':
            raise ValueError('Experimental vLLM backend requires native video mode')
        self.cfg = cfg
        self.processor = AutoProcessor.from_pretrained(cfg.mllm_path)
        self.tokenizer = self.processor.tokenizer
        self.n_calls = 0
        self.last_outputs = []
        self.last_requests = []
        generation_path = Path(cfg.mllm_path)/'generation_config.json'
        defaults = json.loads(generation_path.read_text()) if generation_path.exists() else {}


        self.inherited_top_k = defaults.get('top_k', 50)
        eos = defaults.get('eos_token_id', self.tokenizer.eos_token_id)
        self.stop_token_ids = eos if isinstance(eos, list) else [eos]
        self.engine_settings = dict(
            model=cfg.mllm_path, dtype=cfg.dtype, tensor_parallel_size=1,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len, max_num_seqs=max_num_seqs,
            max_num_batched_tokens=max_model_len, enforce_eager=enforce_eager,
            limit_mm_per_prompt={'image':1, 'video':{'count':1,
                'num_frames':max(cfg.caption_frame_num,cfg.sum_frame_num)}},
            mm_processor_kwargs={'min_pixels':cfg.frame_min_pixels,
                                 'max_pixels':cfg.frame_max_pixels,
                                 'do_resize':False, 'do_sample_frames':False},
            generation_config='vllm', seed=cfg.seed,
            enable_prefix_caching=False,
        )
        self.engine = LLM(**self.engine_settings)

    def _request(self, texts, frames, system):
        frames = list(frames or [])
        req = {'prompt':self._build_prompt(texts, frames, system)}
        if self._uses_video(frames):
            req['multi_modal_data'] = {'video':np.stack([np.asarray(f) for f in frames])}
        elif frames:
            req['multi_modal_data'] = {'image':frames[0]}
        return req

    def _sampling(self, max_new_tokens, temperature, top_p, repetition_penalty, n=1):
        from vllm import SamplingParams

        return SamplingParams(
            n=n, max_tokens=max_new_tokens, temperature=temperature,
            top_p=top_p if temperature > 0 else 1.0,
            top_k=self.inherited_top_k if temperature > 0 else -1,
            repetition_penalty=repetition_penalty,
            stop_token_ids=self.stop_token_ids,


            seed=torch.initial_seed() % (2**32),
        )

    def generate(self, text_blocks, images=None, system=None, max_new_tokens=256,
                 temperature=0.0, top_p=0.9, num_return_sequences=1,
                 repetition_penalty=1.05):
        request = self._request(text_blocks, images, system)
        self.last_requests = [request]
        params = self._sampling(max_new_tokens, temperature, top_p,
                                repetition_penalty, num_return_sequences)
        self.last_outputs = self.engine.generate([request], params, use_tqdm=False)
        self.n_calls += 1
        return [o.text for o in self.last_outputs[0].outputs]

    def generate_batch(self, samples, system=None, max_new_tokens=256,
                       temperature=0.0, top_p=0.9, repetition_penalty=1.05):
        if not samples:
            return []
        requests = [self._request(texts, frames, system) for texts,frames in samples]
        self.last_requests = requests
        params = self._sampling(max_new_tokens, temperature, top_p, repetition_penalty)
        self.last_outputs = self.engine.generate(requests, params, use_tqdm=False)
        self.n_calls += 1
        return [r.outputs[0].text for r in self.last_outputs]
