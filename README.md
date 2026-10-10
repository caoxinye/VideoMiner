# VideoMiner

Inference code for VideoMiner. Model weights and training code are not included.

## Setup

Tested with Python 3.10, CUDA 12.8 and an A800 80 GB GPU.

```bash
pip install -r requirements.txt
```

Set the local model paths in `config.json`:

- `mllm_path`: Qwen2-VL-7B-Instruct
- `judge_llm_path`: Qwen2.5-7B-Instruct
- `judge_lora_path`: policy adapter directory
- `embed_path`: bge-large-en-v1.5

The adapter directory needs `adapter_config.json` and `adapter_model.safetensors`.

## Inference

Run from the repository root:

```bash
CUDA_VISIBLE_DEVICES=0 python run.py \
  --config config.json \
  --video /path/to/video.mp4 \
  --question "What does the person do after closing the door?" \
  --options "Sits down" "Opens a window" "Picks up a bag" "Leaves the room" \
  --ckpt /path/to/policy-checkpoint \
  --out outputs/example.json
```

Uses vLLM by default. Pass `--backend hf` for Hugging Face,
or omit `--options` for an open-ended question. Results are saved as JSON.

Sampling and search parameters, including `fps`, `max_total_frames`, `max_depth`,
`cluster_eps_step`, `judge_temperature` and `sum_frame_num`, are set in the config.
