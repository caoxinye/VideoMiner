import logging
from typing import List, Sequence

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

logger = logging.getLogger(__name__)


class TextEmbedder:
    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
        max_length: int = 512,
        batch_size: int = 16,
    ):
        self.max_length = max_length
        self.batch_size = batch_size
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModel.from_pretrained(model_path, torch_dtype=dtype)
        self.model.eval().to(device)
        self.dim = int(self.model.config.hidden_size)
        logger.info("Loaded text embedder %s (dim=%d)", model_path, self.dim)

    @torch.no_grad()
    def encode(self, texts: Sequence[str]) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]
        texts = [t if t else " " for t in texts]
        out: List[np.ndarray] = []
        for i in range(0, len(texts), self.batch_size):
            batch = list(texts[i : i + self.batch_size])
            enc = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            ).to(self.device)
            hidden = self.model(**enc).last_hidden_state[:, 0]
            hidden = torch.nn.functional.normalize(hidden.float(), p=2, dim=-1)
            out.append(hidden.cpu().numpy())
        return np.concatenate(out, axis=0)

    def encode_one(self, text: str) -> np.ndarray:
        return self.encode([text])[0]


def build_embedder(model_path: str, device: str = "cuda") -> TextEmbedder:

    return TextEmbedder(model_path, device=device)
