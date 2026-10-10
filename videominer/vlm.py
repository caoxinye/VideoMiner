import logging
import re
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor

from . import prompts

logger = logging.getLogger(__name__)

_ACTIONS = ("delete", "stop", "continue")
_PATCH = 28


def _round_to_patch(v: int) -> int:
    return max(_PATCH, int(round(v / _PATCH)) * _PATCH)


def resize_for_vlm(img: Image.Image, max_pixels: int, min_pixels: int) -> Image.Image:

    w, h = img.size
    scale = 1.0
    if w * h > max_pixels:
        scale = (max_pixels / (w * h)) ** 0.5
    elif w * h < min_pixels:
        scale = (min_pixels / (w * h)) ** 0.5
    nw, nh = _round_to_patch(w * scale), _round_to_patch(h * scale)
    return img.resize((nw, nh), Image.BICUBIC)


_MODEL_CLASSES = {
    "qwen2_vl": "Qwen2VLForConditionalGeneration",
    "qwen2_5_vl": "Qwen2_5_VLForConditionalGeneration",
}


def _load_mllm(path: str, dtype, attn_impl: str):


    import transformers
    from transformers import AutoConfig

    model_type = getattr(AutoConfig.from_pretrained(path), "model_type", "")
    cls_name = _MODEL_CLASSES.get(model_type)
    if cls_name is None:
        raise ValueError(
            f"unsupported backbone {model_type!r} at {path}; "
            f"expected one of {sorted(_MODEL_CLASSES)}"
        )
    logger.info("Loading MLLM %s (%s, %s, %s)", path, model_type, dtype, attn_impl)
    cls = getattr(transformers, cls_name)
    return cls.from_pretrained(path, dtype=dtype, attn_implementation=attn_impl)


class QwenVLBackend:
    def __init__(self, cfg):
        self.cfg = cfg
        dtype = getattr(torch, cfg.dtype)
        self.model = _load_mllm(cfg.mllm_path, dtype, cfg.attn_impl)
        self.model.eval().to(cfg.device)
        self.processor = AutoProcessor.from_pretrained(cfg.mllm_path)
        self.tokenizer = self.processor.tokenizer
        self.n_calls = 0


    def _uses_video(self, frames):
        return self.cfg.visual_input_mode == "video" and len(frames) > 1

    def _media_inputs(self, frame_lists):


        images, videos = [], []
        for frames in frame_lists:
            if self._uses_video(frames):
                videos.append(list(frames))
            else:
                images.extend(frames)
        kwargs = {"images": images or None}
        if videos:
            kwargs.update(videos=videos, videos_kwargs={"do_sample_frames": False, "do_resize": False})
        return kwargs

    def _build_prompt(
        self, text_blocks: Sequence[str], images: Sequence[Image.Image], system: Optional[str]
    ) -> str:

        content: List[Dict] = []
        head = list(text_blocks[:-1]) if len(text_blocks) > 1 else []
        for t in head:
            content.append({"type": "text", "text": t})
        content += [{"type": "video"}] if self._uses_video(images) else [{"type": "image"} for _ in images]
        content.append({"type": "text", "text": text_blocks[-1]})

        messages = []
        if system:
            messages.append({"role": "system", "content": [{"type": "text", "text": system}]})
        messages.append({"role": "user", "content": content})
        return self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    @torch.no_grad()
    def generate(
        self,
        text_blocks: Sequence[str],
        images: Optional[Sequence[Image.Image]] = None,
        system: Optional[str] = None,
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        top_p: float = 0.9,
        num_return_sequences: int = 1,
        repetition_penalty: float = 1.05,
    ) -> List[str]:

        images = list(images or [])
        prompt = self._build_prompt(text_blocks, images, system)
        inputs = self.processor(
            text=[prompt],
            **self._media_inputs([images]),
            return_tensors="pt",
            padding=True,
        ).to(self.model.device)

        do_sample = temperature > 0
        out = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature if do_sample else None,
            top_p=top_p if do_sample else None,
            num_return_sequences=num_return_sequences,
            repetition_penalty=repetition_penalty,
            pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
        )
        trimmed = out[:, inputs.input_ids.shape[1] :]
        self.n_calls += 1
        return self.processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

    @torch.no_grad()
    def generate_batch(
        self,
        samples: Sequence[Tuple[Sequence[str], Sequence[Image.Image]]],
        system: Optional[str] = None,
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        top_p: float = 0.9,
        repetition_penalty: float = 1.05,
    ) -> List[str]:

        if not samples:
            return []
        prompts_text = [self._build_prompt(t, i, system) for t, i in samples]

        prev_side = self.tokenizer.padding_side
        self.tokenizer.padding_side = "left"
        try:
            inputs = self.processor(
                text=prompts_text,
                **self._media_inputs([imgs for _, imgs in samples]),
                return_tensors="pt",
                padding=True,
            ).to(self.model.device)

            do_sample = temperature > 0
            out = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature if do_sample else None,
                top_p=top_p if do_sample else None,
                repetition_penalty=repetition_penalty,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            )
        finally:
            self.tokenizer.padding_side = prev_side

        trimmed = out[:, inputs.input_ids.shape[1] :]
        self.n_calls += 1
        return self.processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

    def prep_frames(self, frames: Sequence[Image.Image]) -> List[Image.Image]:
        return [
            resize_for_vlm(f, self.cfg.frame_max_pixels, self.cfg.frame_min_pixels)
            for f in frames
        ]


    def _caption_body(self, question: str, root_caption: str, max_words: int) -> str:
        if self.cfg.caption_prompt_style == "compact":
            context = ("Background only; include it only when visible here:\n" + root_caption
                       if self.cfg.caption_use_root_context and root_caption else "")
            return prompts.CAPTION_USER_COMPACT.format(
                question=question if self.cfg.caption_use_query and question else "No specific question; record the observed events.",
                context=context, max_words=max_words,
            )
        if self.cfg.caption_prompt_style == "structured":
            body = prompts.CAPTION_USER_STRUCTURED.format(
                question=question if self.cfg.caption_use_query and question else "No specific question; record the observed events.",
                max_words=max_words,
            )
        elif self.cfg.caption_prompt_style != "legacy":
            raise ValueError(f"Unknown caption prompt style: {self.cfg.caption_prompt_style}")
        elif self.cfg.caption_use_query and question:
            body = prompts.CAPTION_USER_WITH_QUERY.format(
                question=question, max_words=max_words
            )
        else:
            body = prompts.CAPTION_USER_NO_QUERY.format(max_words=max_words)
        if self.cfg.caption_use_root_context and root_caption:
            if self.cfg.caption_context_style == "description_prefix":
                description = re.split(r"\bEVIDENCE\s*:", root_caption, maxsplit=1,
                                       flags=re.IGNORECASE)[0].strip()
                body = prompts.CAPTION_DESCRIPTION_CONTEXT.format(description=description) + body
                body += "\nDescribe only events visible in the currently supplied span. Do not copy the background summary."
            elif self.cfg.caption_context_style == "legacy":
                body += prompts.CAPTION_ROOT_CONTEXT.format(root_caption=root_caption)
            else:
                raise ValueError(f"Unknown caption context style: {self.cfg.caption_context_style}")
        return body

    def caption(
        self,
        frames: Sequence[Image.Image],
        question: str = "",
        root_caption: str = "",
        max_words: Optional[int] = None,
    ) -> str:
        body = self._caption_body(question, root_caption, self.cfg.caption_max_words if max_words is None else max_words)
        res = self.generate(
            [body],
            images=self.prep_frames(frames),
            system=prompts.CAPTION_SYSTEM,
            max_new_tokens=self.cfg.caption_max_new_tokens,
            temperature=self.cfg.caption_temperature,
        )[0]
        return res.strip()

    def caption_batch(
        self,
        frame_lists: Sequence[Sequence[Image.Image]],
        question: str = "",
        root_caption: str = "",
        max_words: Optional[int] = None,
    ) -> List[str]:

        body = self._caption_body(question, root_caption, self.cfg.caption_max_words if max_words is None else max_words)
        samples = [([body], self.prep_frames(f)) for f in frame_lists]
        outs = self.generate_batch(
            samples,
            system=prompts.CAPTION_SYSTEM,
            max_new_tokens=self.cfg.caption_max_new_tokens,
            temperature=self.cfg.caption_temperature,
        )
        return [o.strip() for o in outs]


    def judge(
        self, question: str, caption: str, position: int, root_caption: str = "",
        coverage: str = "",
    ) -> Tuple[str, str, Dict[str, float]]:

        prompt = prompts.JUDGE_PROMPT.format(
            question=question,
            caption=caption,
            position=position,
            max_depth=self.cfg.max_depth,
            root_caption=root_caption or "(not available)",
            coverage=coverage or "(unknown)",
        )
        k = max(1, self.cfg.judge_samples)
        outs = self.generate(
            [prompt],
            images=None,
            system=prompts.JUDGE_SYSTEM,
            max_new_tokens=self.cfg.judge_max_new_tokens,
            temperature=self.cfg.judge_temperature if k > 1 else 0.0,
            top_p=self.cfg.judge_top_p,
            num_return_sequences=k,
            repetition_penalty=1.1,
        )
        votes = {a: 0.0 for a in _ACTIONS}
        traces = []
        for o in outs:
            a = parse_action(o)
            traces.append(o.strip())
            if a in votes:
                votes[a] += 1.0
        total = sum(votes.values())
        if total == 0:

            return "continue", traces[0] if traces else "", {"continue": 1.0, "stop": 0.0, "delete": 0.0}
        votes = {a: v / total for a, v in votes.items()}
        action = max(votes, key=votes.get)
        best = next((t for t in traces if parse_action(t) == action), traces[0])
        return action, best, votes


    def answer(
        self,
        frames: Sequence[Image.Image],
        question: str,
        captions: Sequence[str] = (),
        options: str = "",
        letters: str = "A, B, C, D",
    ) -> str:
        if self.cfg.answer_prompt_style == "reference":


            context = "Please answer the question based on the video content."
            if self.cfg.use_captions and captions:
                joined = "\n".join(f"{i + 1}. {c}" for i, c in enumerate(captions) if c)
                context = "Please answer based on captions and video. Captions:\n" + joined
            instruction = (f"Answer with a single letter from {letters}. The answer is:"
                           if options else "Answer concisely.")
            tail = f"{question}\n{options}\n{instruction}"
            imgs = self.prep_frames(frames) if (self.cfg.use_key_frames and frames) else []
            return self.generate(
                [context, tail], images=imgs, system=None,
                max_new_tokens=self.cfg.answer_max_new_tokens,
                temperature=self.cfg.answer_temperature,
            )[0].strip()

        caption_block = ""
        if self.cfg.use_captions and captions:
            joined = "\n".join(f"{i + 1}. {c}" for i, c in enumerate(captions) if c)
            template = prompts.ANSWER_TIMED_CAPTION_BLOCK if self.cfg.caption_timestamps else prompts.ANSWER_CAPTION_BLOCK
            caption_block = template.format(captions=joined)

        if options:
            tpl = prompts.ANSWER_MC_THINK if self.cfg.answer_think else prompts.ANSWER_MC_DIRECT
            body = tpl.format(
                caption_block=caption_block,
                question=question,
                options=options,
                letters=letters,
            )
        else:
            body = prompts.ANSWER_OPEN.format(
                caption_block=caption_block, question=question
            )

        imgs = self.prep_frames(frames) if (self.cfg.use_key_frames and frames) else []
        if self._uses_video(imgs):
            body = body.replace("The images above are the key frames", "The video sequence above contains the key frames")

        head, tail = (body.split("\n\nQuestion:", 1) + [""])[:2]
        blocks = [head, "Question:" + tail] if tail else [body]
        return self.generate(
            blocks,
            images=imgs,
            system=prompts.ANSWER_SYSTEM,
            max_new_tokens=self.cfg.answer_max_new_tokens,
            temperature=self.cfg.answer_temperature,
        )[0].strip()


def parse_action(text: str) -> str:
    m = re.search(r"<action>(.*?)</action>", text, re.DOTALL | re.IGNORECASE)
    if not m:
        m = re.search(r"\[?action\]?\s*[:>]?\s*(delete|stop|accept|continue)\b", text, re.IGNORECASE)
    if not m:

        tail = re.split(r"</think>", text, flags=re.IGNORECASE)[-1]
        found = re.findall(r"\b(delete|stop|accept|continue)\b", tail, re.IGNORECASE)
        action = found[-1].lower() if found else ""
    else:
        action = m.group(1).strip().lower()
    return "stop" if action == "accept" else action if action in _ACTIONS else ""


def parse_choice(text: str, letters: Sequence[str]) -> str:

    up = set(letters)
    m = re.search(r"<answer>\s*([A-Za-z])\s*</answer>", text, re.IGNORECASE)
    if m and m.group(1).upper() in up:
        return m.group(1).upper()
    tail = re.split(r"</think>", text, flags=re.IGNORECASE)[-1]
    m = re.search(r"(?:answer|choice)\s*(?:is|:)\s*\(?([A-Za-z])\)?\b", tail, re.IGNORECASE)
    if m and m.group(1).upper() in up:
        return m.group(1).upper()
    m = re.search(r"\b(?:is|option|choice)\s*\(?([A-Z])\)?[.。]?\s*$", tail)
    if m and m.group(1) in up:
        return m.group(1)

    m = re.fullmatch(r"\s*\(?([A-Za-z])\)?[.。]?\s*", tail)
    if m and m.group(1).upper() in up:
        return m.group(1).upper()
    return ""
