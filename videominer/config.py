from dataclasses import dataclass, asdict
from typing import Optional


@dataclass
class VideoMinerConfig:

    mode: str = 'videominer'


    mllm_path: str = 'models/Qwen2-VL-7B-Instruct'
    embed_path: str = 'models/bge-large-en-v1.5'
    attn_impl: str = 'sdpa'
    dtype: str = 'bfloat16'
    device: str = 'cuda'
    visual_input_mode: str = 'video'


    fps: float = 1.0
    max_total_frames: int = 512
    frame_max_pixels: int = 200704
    frame_min_pixels: int = 50176


    max_depth: int = 4
    max_seg_num: int = 4
    min_seg_frames: int = 16
    caption_frame_num: int = 32
    use_cluster: bool = True
    dbscan_eps: float = 0.15
    dbscan_min_pts: int = 1
    cluster_policy: str = 'adaptive_reference'
    cluster_eps_start: float = 0.01
    cluster_eps_step: float = 0.001


    delta_delete: float = 1.0
    delta_accept: float = 1.0
    delta_continue: float = 1.0
    adapt_depth_by_auxin: bool = True
    continue_vote_base: float = 0.5


    judge_backend: str = 'lora_llm'
    judge_llm_path: str = 'models/Qwen2.5-7B-Instruct'
    judge_lora_path: str = 'models/policy-checkpoint'
    judge_device: str = ''
    judge_prompt_style: str = 'action_only'
    judge_use_options: bool = True
    judge_samples: int = 1
    judge_temperature: float = 0.7
    judge_top_p: float = 0.92
    judge_max_new_tokens: int = 320
    force_root_continue: bool = True


    caption_temperature: float = 0.2
    caption_max_new_tokens: int = 512
    caption_use_query: bool = True
    caption_use_options: bool = True
    caption_use_root_context: bool = False
    caption_context_style: str = 'legacy'
    caption_prompt_style: str = 'compact'
    caption_max_words: int = 200


    sum_frame_num: int = 32
    frame_allocation: str = 'length'
    answer_max_new_tokens: int = 512
    answer_temperature: float = 0.0
    answer_prompt_style: str = 'grounded'
    use_key_frames: bool = True


    also_frames_only: bool = False
    also_uniform: bool = False
    also_mined_frames: bool = False
    use_captions: bool = False
    answer_include_root_caption: bool = True
    caption_timestamps: bool = False
    answer_think: bool = False


    seed: int = 42
    log_level: str = 'INFO'
    dump_tree: bool = True

    def to_dict(self):
        return asdict(self)

    @property
    def lambda_auxin(self) -> float:
        return (self.delta_delete + self.delta_accept) / (2.0 * self.delta_continue)

    @property
    def effective_max_depth(self) -> int:

        if not self.adapt_depth_by_auxin:
            return self.max_depth
        d = round(self.max_depth / max(self.lambda_auxin, 1e-6))
        return int(max(2, min(d, self.max_depth + 2)))

    def continue_threshold(self, depth: int) -> float:

        tau = self.continue_vote_base * (self.lambda_auxin ** max(depth - 1, 0))
        return float(min(0.95, max(0.0, tau)))
