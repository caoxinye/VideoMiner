import json
import logging
import os
import time
from typing import Dict, List, Optional, Sequence

from .config import VideoMinerConfig
from .embedder import build_embedder
from .judge import build_judge
from .tree import VideoTree, collect_key_nodes, select_key_frames, selection_variants
from .video import load_frames, sample_uniform
from .vlm import QwenVLBackend, parse_choice

logger = logging.getLogger(__name__)

DEFAULT_LETTERS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")


def format_options(options: Sequence[str], letters: Sequence[str]) -> str:
    return "\n".join(f"{l}. {o}" for l, o in zip(letters, options))


def timestamped_caption(node, stamps):

    if not stamps or not node.frame_ids:
        return node.caption
    intervals = []
    first = previous = node.frame_ids[0]
    for i in node.frame_ids[1:]:
        if i != previous + 1:
            intervals.append((first, previous))
            first = i
        previous = i
    intervals.append((first, previous))
    times = ", ".join(f"{stamps[a]:.1f}-{stamps[b]:.1f}s" for a, b in intervals)
    return f"[Sampled evidence intervals: {times}] {node.caption}"


class VideoMiner:
    def __init__(self, cfg: Optional[VideoMinerConfig] = None, backend=None, embedder=None, judge=None):
        self.cfg = cfg or VideoMinerConfig()
        self.backend = backend or QwenVLBackend(self.cfg)
        self.embedder = embedder or build_embedder(self.cfg.embed_path, self.cfg.device)
        self.judge = judge or build_judge(self.cfg, self.backend)
        logger.info(
            "VideoMiner ready | judge=%s lambda_auxin=%.3f effective_max_depth=%d",
            self.cfg.judge_backend, self.cfg.lambda_auxin, self.cfg.effective_max_depth,
        )

    def run(
        self,
        video_path: str,
        question: str,
        options: Optional[Sequence[str]] = None,
        letters: Optional[Sequence[str]] = None,
        frames: Optional[Sequence] = None,
        stamps: Optional[Sequence[float]] = None,
    ) -> Dict:
        cfg = self.cfg
        t0 = time.time()
        if frames is None:
            frames, stamps = load_frames(video_path, cfg.fps, cfg.max_total_frames)
        frames, stamps = list(frames), list(stamps or [])
        if not frames:
            raise RuntimeError(f"no frames decoded from {video_path}")
        logger.info("decoded %d frames (%.1f s of video)", len(frames), stamps[-1] if stamps else 0)

        letters = list(letters if letters is not None else DEFAULT_LETTERS[:len(options or [])])
        if options and (len(letters) != len(options) or len(set(letters)) != len(letters)):
            raise ValueError("Each option requires a unique answer letter (up to 26 options)")
        opt_block = format_options(options, letters) if options else ""
        selection_fallback = ""

        if cfg.mode == "uniform":
            key_ids = sample_uniform(list(range(len(frames))), cfg.sum_frame_num)
            key_nodes, captions, tree_builder, root = [], [], None, None
        else:
            full_query = question if not opt_block else f"{question}\nOptions:\n{opt_block}"
            tree_query = full_query if cfg.judge_use_options else question

            tree_builder = VideoTree(cfg, self.backend, self.embedder, self.judge)


            caption_query = full_query if cfg.caption_use_options else question
            root = tree_builder.build(frames, tree_query, stamps, caption_question=caption_query)

            key_nodes = collect_key_nodes(root)
            if len(key_nodes) == 1 and key_nodes[0] is root and root.state != "stop":
                selection_fallback = "whole_video_no_accepted_nodes"
                logger.warning("No accepted event survived; using whole-video fallback")
            key_ids = select_key_frames(key_nodes, cfg.sum_frame_num, len(frames),
                                       allocation=cfg.frame_allocation)
            caption_nodes = [n for n in key_nodes if n.caption]
            if cfg.answer_include_root_caption and root not in key_nodes and root.caption:
                caption_nodes = [root] + caption_nodes
            captions = [timestamped_caption(n, stamps) if cfg.caption_timestamps else n.caption
                        for n in caption_nodes]
            logger.info("key nodes: %d | key frames: %d", len(key_nodes), len(key_ids))

        raw = self.backend.answer(
            [frames[i] for i in key_ids],
            question=question,
            captions=captions,
            options=opt_block,
            letters=", ".join(letters),
        )
        pred = parse_choice(raw, letters) if letters else raw


        variant_preds: Dict[str, str] = {}
        if cfg.also_mined_frames and root is not None:
            raw_mined = self.backend.answer(
                [frames[i] for i in key_ids], question=question, captions=(),
                options=opt_block, letters=", ".join(letters),
            )
            variant_preds["keynode"] = parse_choice(raw_mined, letters) if letters else raw_mined
        if cfg.also_uniform and root is not None:
            ids = sample_uniform(list(range(len(frames))), cfg.sum_frame_num)
            raw_uniform = self.backend.answer(
                [frames[i] for i in ids], question=question, captions=(),
                options=opt_block, letters=", ".join(letters),
            )
            variant_preds["uniform"] = parse_choice(raw_uniform, letters) if letters else raw_uniform
        if cfg.also_frames_only and root is not None:
            for name, ids in selection_variants(key_nodes, cfg.sum_frame_num, len(frames)).items():
                raw_v = self.backend.answer(
                    [frames[i] for i in ids],
                    question=question,
                    captions=(),
                    options=opt_block,
                    letters=", ".join(letters),
                )
                variant_preds[name] = parse_choice(raw_v, letters) if letters else raw_v

        result = {
            "video": video_path,
            "mode": cfg.mode,
            "question": question,
            "options": list(options) if options else [],
            "pred": pred,
            "raw_answer": raw,
            "n_frames_sampled": len(frames),
            "n_key_nodes": len(key_nodes),
            "selection_fallback": selection_fallback,
            "key_frame_ids": key_ids,
            "key_frame_timestamps": [round(stamps[i], 2) for i in key_ids if i < len(stamps)],
            "root_caption": root.caption if root else "",
            "captions": captions,
            "n_caption_calls": tree_builder.n_caption_calls if tree_builder else 0,
            "n_judge_calls": tree_builder.n_judge_calls if tree_builder else 0,
            "elapsed_s": round(time.time() - t0, 1),
        }
        if cfg.dump_tree and root is not None:
            result["tree"] = root.to_dict()
        if variant_preds:
            result["variant_preds"] = variant_preds
            if "keynode" in variant_preds:
                result["pred_frames_only"] = variant_preds["keynode"]
        return result


def save_jsonl(path: str, records: List[Dict]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
