from typing import List, Sequence, Optional

import numpy as np


def gray_histograms(frames: Sequence["np.ndarray"], bins: int = 256) -> np.ndarray:

    hists = np.empty((len(frames), bins), dtype=np.float64)
    for i, img in enumerate(frames):
        arr = np.asarray(img)
        if arr.ndim == 3:

            arr = arr[..., 0] * 0.299 + arr[..., 1] * 0.587 + arr[..., 2] * 0.114
        h, _ = np.histogram(arr, bins=bins, range=(0, 256))
        s = h.sum()
        hists[i] = h / s if s > 0 else 1.0 / bins
    return hists


def bhattacharyya_distances(hists: np.ndarray) -> np.ndarray:

    if len(hists) < 2:
        return np.zeros(0, dtype=np.float64)
    bc = np.sqrt(hists[:-1] * hists[1:]).sum(axis=1)
    return -np.log(np.clip(bc, 1e-12, 1.0))


def split_by_change_points(
    indices: List[int],
    diffs: np.ndarray,
    max_seg_num: int,
    min_seg_frames: int,
    histograms: Optional[np.ndarray] = None,
) -> List[List[int]]:


    n = len(indices)
    if n < 2 or max_seg_num < 2:
        return [list(indices)]

    min_len = max(1, min(min_seg_frames // max_seg_num, n // max_seg_num))

    if histograms is not None:


        cand = bhattacharyya_distances(histograms[indices])
    else:
        if any(b != a + 1 for a, b in zip(indices, indices[1:])):
            raise ValueError("Non-contiguous events require their frame histograms")
        cand = np.asarray([diffs[i] for i in indices[:-1]], dtype=np.float64)
    order = np.argsort(-cand)

    cuts: List[int] = []
    for j in order:
        if len(cuts) >= max_seg_num - 1:
            break
        j = int(j)
        bounds = sorted(cuts + [j])
        starts = [-1] + bounds + [n - 1]
        if all(starts[k + 1] - starts[k] >= min_len for k in range(len(starts) - 1)):
            cuts.append(j)
    cuts.sort()

    segs, prev = [], 0
    for c in cuts:
        segs.append(list(indices[prev : c + 1]))
        prev = c + 1
    segs.append(list(indices[prev:]))
    return [s for s in segs if s]
