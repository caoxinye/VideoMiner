from typing import List, Sequence

import numpy as np


def cosine_distance_matrix(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    x = x / np.clip(norm, 1e-12, None)
    return np.clip(1.0 - x @ x.T, 0.0, 2.0)


def dbscan(dist: np.ndarray, eps: float, min_pts: int) -> np.ndarray:

    n = dist.shape[0]
    labels = np.full(n, -1, dtype=int)
    visited = np.zeros(n, dtype=bool)
    neigh = [np.where(dist[i] <= eps)[0] for i in range(n)]
    cid = 0
    for i in range(n):
        if visited[i]:
            continue
        visited[i] = True
        if len(neigh[i]) < min_pts:
            continue
        labels[i] = cid
        queue = [j for j in neigh[i] if j != i]
        while queue:
            j = queue.pop(0)
            if labels[j] == -1:
                labels[j] = cid
            if not visited[j]:
                visited[j] = True
                if len(neigh[j]) >= min_pts:
                    queue.extend(k for k in neigh[j] if labels[k] == -1)
        cid += 1
    return labels


def cluster_segments(
    embeddings: Sequence[np.ndarray], eps: float, min_pts: int
) -> List[List[int]]:


    n = len(embeddings)
    if n == 0:
        return []
    if n == 1:
        return [[0]]
    labels = dbscan(cosine_distance_matrix(np.stack(embeddings)), eps, min_pts)
    groups: dict = {}
    for i, lab in enumerate(labels):
        key = f"noise_{i}" if lab == -1 else int(lab)
        groups.setdefault(key, []).append(i)

    return sorted(groups.values(), key=lambda g: min(g))


def cluster_segments_adaptive(embeddings, min_pts=1, start=0.01, step=0.01):


    if not np.isfinite(start) or not np.isfinite(step) or not 0 < start <= 2 or step <= 0:
        raise ValueError('Adaptive DBSCAN needs a positive radius and step')
    n = len(embeddings)
    if n == 0:
        return [], start
    if not 1 <= min_pts <= n:
        raise ValueError('min_pts must fit the number of events')
    distances = cosine_distance_matrix(np.stack(embeddings))
    if not np.isfinite(distances).all():
        raise ValueError('Non-finite caption embeddings')
    for index in range(int(np.ceil((2-start)/step))+1):
        radius = min(2., start+index*step)
        labels = dbscan(distances, radius, min_pts)
        groups = {}
        for i,label in enumerate(labels):
            key = ('noise',i) if label == -1 else ('cluster',int(label))
            groups.setdefault(key,[]).append(i)
        result = sorted(groups.values(),key=lambda group:group[0])
        if len(result) < n or n == 1:
            return result, radius
    raise RuntimeError('Adaptive DBSCAN did not merge within the cosine-distance range')
