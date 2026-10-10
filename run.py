import argparse
import hashlib
import json
import logging
import os
from pathlib import Path

from videominer.config import VideoMinerConfig


def load_config(path, ckpt=None):
    cfg = VideoMinerConfig(**json.loads(Path(path).read_text()))
    if ckpt is not None:
        cfg.judge_lora_path = ckpt
    if not cfg.judge_lora_path.strip():
        raise ValueError("Provide a policy adapter with --ckpt or judge_lora_path")
    if cfg.sum_frame_num != 32 or cfg.judge_samples != 1:
        raise ValueError("This version uses 32 answer frames and one policy action per node")
    if (cfg.delta_delete, cfg.delta_accept, cfg.delta_continue) != (1, 1, 1):
        raise ValueError("Keep the growth parameters neutral")
    return cfg


def check_files(cfg, video):
    if not Path(video).is_file():
        raise FileNotFoundError(video)
    for key in ("mllm_path", "judge_llm_path", "embed_path", "judge_lora_path"):
        directory = Path(getattr(cfg, key)).expanduser()
        setattr(cfg, key, str(directory))
        names = ("adapter_config.json", "adapter_model.safetensors") if key == "judge_lora_path" else ("config.json",)
        for name in names:
            if not (directory / name).is_file():
                raise FileNotFoundError(directory / name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--question", required=True)
    parser.add_argument("--options", nargs="+")
    parser.add_argument("--ckpt")
    parser.add_argument("--config", default=str(Path(__file__).with_name("config.json")))
    parser.add_argument("--backend", choices=["vllm", "hf"], default="vllm")
    parser.add_argument("--uid", default="single")
    parser.add_argument("--out", default="outputs/result.json")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    if args.options and not 2 <= len(args.options) <= 26:
        parser.error("Expected 2–26 answer options")
    cfg = load_config(args.config, args.ckpt)
    check_files(cfg, args.video)
    if args.check:
        print("Input and model paths OK")
        return
    output = Path(args.out)
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if args.backend == "vllm":
        os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
        os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    from transformers import set_seed
    from videominer.pipeline import VideoMiner

    set_seed(cfg.seed)
    backend = None
    try:
        if args.backend == "vllm":
            from videominer.vllm_backend import VLLMBackend
            backend = VLLMBackend(cfg)
        miner = VideoMiner(cfg, backend=backend)
        uid_hash = hashlib.sha256(json.dumps(args.uid, ensure_ascii=False).encode()).hexdigest()
        seed = (cfg.seed + int(uid_hash[:8], 16)) % (2 ** 32)
        set_seed(seed)
        result = miner.run(args.video, args.question, args.options)
        result.update(uid=args.uid, seed=seed, config=cfg.to_dict(), backend=args.backend)
        with output.open("x") as handle:
            json.dump(result, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        print(f"Answer: {result['pred']}")
        print(f"Saved to {output}")
    finally:
        if backend is not None:
            backend.close()


if __name__ == "__main__":
    main()
