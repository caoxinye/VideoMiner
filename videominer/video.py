import logging
from typing import List, Sequence, Tuple

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


def _uniform_indices(n_total: int, fps_native: float, fps_target: float, cap: int, min_frames: int = 32) -> List[int]:
    if cap < 1 or fps_target <= 0 or fps_native <= 0:
        raise ValueError("Frame cap and frame rates must be positive")
    if n_total <= 0:
        return []
    step = max(1.0, fps_native / max(fps_target, 1e-6))
    idx = np.arange(0, n_total, step).astype(int)
    if len(idx) > cap:
        idx = idx[np.linspace(0, len(idx) - 1, cap).astype(int)]
    minimum = min(n_total, cap, min_frames)
    if len(idx) < minimum:
        idx = np.linspace(0, n_total - 1, minimum).astype(int)
    if len(idx) == 0:
        idx = np.array([0])
    return sorted(set(int(i) for i in idx))


def load_frames(
    video_path: str, fps: float, max_total_frames: int
) -> Tuple[List[Image.Image], List[float]]:

    try:
        import decord

        decord.bridge.set_bridge("native")
        vr = decord.VideoReader(video_path, num_threads=4)
        native = float(vr.get_avg_fps()) or 30.0
        idx = _uniform_indices(len(vr), native, fps, max_total_frames)
        batch = vr.get_batch(idx).asnumpy()
        frames = [Image.fromarray(a) for a in batch]
        return frames, [i / native for i in idx]
    except Exception as exc:
        logger.warning("decord failed (%s: %s), falling back to cv2", type(exc).__name__, exc)

    import cv2

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {video_path}")
    native = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idx = _uniform_indices(n_total, native, fps, max_total_frames)
    frames, stamps = [], []
    for i in idx:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, img = cap.read()
        if not ok:
            continue
        frames.append(Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))
        stamps.append(i / native)
    cap.release()
    return frames, stamps


def sample_uniform(items: Sequence, n: int) -> List:

    if n <= 0:
        return []
    if len(items) <= n:
        return list(items)
    idx = np.linspace(0, len(items) - 1, n).astype(int)
    return [items[i] for i in idx]
