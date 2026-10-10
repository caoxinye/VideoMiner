import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

from .cluster import cluster_segments, cluster_segments_adaptive
from .scene import bhattacharyya_distances, gray_histograms, split_by_change_points
from .video import sample_uniform

logger = logging.getLogger(__name__)


@dataclass
class TreeNode:
    frame_ids: List[int]
    depth: int = 1
    caption: str = ""
    embedding: Optional[np.ndarray] = None
    state: str = "continue"
    analysis: str = ""
    votes: Dict[str, float] = field(default_factory=dict)
    relevance: float = 0.0
    stop_conf: float = 0.0
    children: List["TreeNode"] = field(default_factory=list)
    parent: Optional["TreeNode"] = None
    segment_count: int = 0
    cluster_count: int = 0
    cluster_radius: Optional[float] = None
    stop_reason: str = ""

    def add_child(self, child: "TreeNode") -> None:
        child.parent = self
        child.depth = self.depth + 1
        self.children.append(child)

    @property
    def n_frames(self) -> int:
        return len(self.frame_ids)

    def to_dict(self) -> Dict:
        return {
            "depth": self.depth,
            "n_frames": self.n_frames,
            "frame_ids": self.frame_ids,
            "frame_span": [self.frame_ids[0], self.frame_ids[-1]] if self.frame_ids else [],
            "state": self.state,
            "votes": self.votes,
            "relevance": round(self.relevance, 4),
            "stop_conf": round(self.stop_conf, 4),
            "caption": self.caption,
            "analysis": self.analysis,
            "segment_count": self.segment_count,
            "cluster_count": self.cluster_count,
            "cluster_radius": self.cluster_radius,
            "stop_reason": self.stop_reason,
            "children": [c.to_dict() for c in self.children],
        }


class VideoTree:
    def __init__(self, cfg, backend, embedder, judge=None):
        self.cfg = cfg
        self.backend = backend
        self.embedder = embedder
        self.judge = judge if judge is not None else backend
        self.root_caption = ""
        self.n_caption_calls = 0
        self.n_judge_calls = 0


    def build(self, frames: Sequence, question: str, timestamps: Optional[Sequence[float]] = None,
              *, caption_question: Optional[str] = None) -> TreeNode:
        cfg = self.cfg
        self.frames = list(frames)
        self.timestamps = list(timestamps) if timestamps else []
        self.histograms = gray_histograms(self.frames)
        self.diffs = bhattacharyya_distances(self.histograms)
        self.question = question
        self.caption_question = question if caption_question is None else caption_question
        self.question_embedding = self.embedder.encode_one(question)
        self.root_caption = ""

        root = TreeNode(frame_ids=list(range(len(self.frames))), depth=1)
        self._dfs(root)
        _mark_leaves(root)
        logger.info(
            "tree built: %d nodes, %d caption calls, %d judge calls",
            _count(root), self.n_caption_calls, self.n_judge_calls,
        )
        return root

    def _coverage(self, node: TreeNode) -> str:
        n = node.n_frames
        if not self.timestamps or not node.frame_ids:
            return f"{n} sampled frames"
        span = self.timestamps[node.frame_ids[-1]] - self.timestamps[node.frame_ids[0]]
        total = self.timestamps[-1] - self.timestamps[0]
        share = 100.0 * span / total if total > 0 else 100.0
        return f"about {span:.0f} s of video ({share:.0f}% of the whole video), {n} sampled frames"

    def _caption(self, frame_ids: Sequence[int]) -> str:
        picked = sample_uniform(list(frame_ids), self.cfg.caption_frame_num)
        self.n_caption_calls += 1
        return self.backend.caption(
            [self.frames[i] for i in picked],
            question=self.caption_question,
            root_caption=self.root_caption,
        )

    def _caption_batch(self, spans: Sequence[Sequence[int]]) -> List[str]:

        frame_lists = [
            [self.frames[i] for i in sample_uniform(list(s), self.cfg.caption_frame_num)]
            for s in spans
        ]
        self.n_caption_calls += len(frame_lists)
        return self.backend.caption_batch(
            frame_lists, question=self.caption_question, root_caption=self.root_caption
        )

    def _dfs(
        self,
        node: TreeNode,
        caption: Optional[str] = None,
        embedding: Optional[np.ndarray] = None,
    ) -> None:
        cfg = self.cfg
        node.caption = caption if caption is not None else self._caption(node.frame_ids)
        if node.parent is None:
            self.root_caption = node.caption
        node.embedding = (
            embedding if embedding is not None else self.embedder.encode_one(node.caption)
        )

        self.n_judge_calls += 1
        action, analysis, votes = self.judge.judge(
            self.question, node.caption, node.depth, self.root_caption, self._coverage(node)
        )
        node.analysis, node.votes = analysis, votes
        node.stop_conf = float(votes.get("stop", 0.0))
        node.relevance = float(
            np.dot(node.embedding, self.question_embedding)
            / max(np.linalg.norm(node.embedding) * np.linalg.norm(self.question_embedding), 1e-12)
        )


        tau = cfg.continue_threshold(node.depth)
        if action == "continue" and votes.get("continue", 0.0) < tau:
            action = "stop" if votes.get("stop", 0.0) >= votes.get("delete", 0.0) else "delete"
            logger.debug("auxin downgrade at depth %d (tau=%.2f) -> %s", node.depth, tau, action)
        node.state = action

        if node.parent is None and cfg.force_root_continue and node.state != "continue":
            node.state = "continue"

        if node.state in ("stop", "delete"):
            return
        if node.depth >= cfg.effective_max_depth:
            return
        if node.n_frames <= cfg.min_seg_frames:
            return

        segs = split_by_change_points(
            node.frame_ids, self.diffs, cfg.max_seg_num, cfg.min_seg_frames,
            histograms=self.histograms,
        )
        if len(segs) < 2:
            return

        seg_caps = self._caption_batch(segs)
        seg_embs = list(self.embedder.encode(seg_caps))
        node.segment_count = len(segs)

        if cfg.use_cluster:
            if cfg.cluster_policy == "adaptive_reference":
                groups, node.cluster_radius = cluster_segments_adaptive(
                    seg_embs, cfg.dbscan_min_pts, cfg.cluster_eps_start, cfg.cluster_eps_step)
            elif cfg.cluster_policy == "fixed":
                groups = cluster_segments(seg_embs, cfg.dbscan_eps, cfg.dbscan_min_pts)
                node.cluster_radius = cfg.dbscan_eps
            else:
                raise ValueError(f"Unknown cluster policy: {cfg.cluster_policy}")
            node.cluster_count = len(groups)
            if len(groups) < 2 and cfg.cluster_policy == "adaptive_reference":
                node.stop_reason = "single_semantic_cluster"
                return
            if len(groups) < 2:

                groups = [[i] for i in range(len(segs))]
        else:
            groups = [[i] for i in range(len(segs))]

        for g in groups:
            ids = sorted({fid for i in g for fid in segs[i]})
            child = TreeNode(frame_ids=ids)
            node.add_child(child)
            if len(g) == 1:

                self._dfs(child, seg_caps[g[0]], seg_embs[g[0]])
            else:

                self._dfs(child)


def _mark_leaves(node: TreeNode) -> None:
    if not node.children and node.state != "delete":
        node.state = "stop"
    for c in node.children:
        _mark_leaves(c)


def _count(node: TreeNode) -> int:
    return 1 + sum(_count(c) for c in node.children)


def collect_key_nodes(root: TreeNode) -> List[TreeNode]:


    out: List[TreeNode] = []

    def walk(n: TreeNode) -> None:
        if n.state == "stop":
            out.append(n)
        elif n.state == "continue":
            for child in n.children:
                walk(child)

    walk(root)
    if not out:
        out.append(root)
    return sorted(out, key=lambda n: n.frame_ids[0] if n.frame_ids else 0)


def select_key_frames(
    nodes: Sequence[TreeNode],
    budget: int,
    n_total: Optional[int] = None,
    weights: Optional[Sequence[float]] = None,
    allocation: str = "length",
) -> List[int]:


    nodes = [n for n in nodes if n.n_frames]
    if allocation not in ("length", "event_equal", "temporal_anchor", "accepted_union", "guarded_anchor"):
        raise ValueError(f"Unknown frame allocation: {allocation}")
    if allocation == "guarded_anchor":
        if weights is not None or not n_total or n_total < 1:
            raise ValueError("Guarded anchors require n_total and default weights")
        accepted = {i for node in nodes for i in node.frame_ids}
        if accepted == set(range(n_total)):
            return sample_uniform(list(range(n_total)), budget)
        return select_key_frames(nodes, budget, n_total, allocation="temporal_anchor")
    if allocation == "accepted_union":
        if weights is not None:
            raise ValueError("Accepted-union sampling does not use node weights")
        accepted = sorted({i for node in nodes for i in node.frame_ids})
        picked = sample_uniform(accepted, budget)
        if picked and n_total and len(picked) < min(budget, n_total):
            picked = _dilate(picked, min(budget, n_total), n_total)
        return picked
    if allocation == "temporal_anchor":
        if weights is not None or not n_total or n_total < 1:
            raise ValueError("Temporal anchors require n_total and default weights")
        control = select_key_frames(nodes, budget, n_total)
        uniform = sample_uniform(list(range(n_total)), budget)


        if not control or control == uniform or budget <= 0:
            return control
        anchors = sample_uniform(list(range(n_total)), max(1, min(budget, n_total) // 4))
        remaining = [i for i in control if i not in set(anchors)]
        selected = sorted(set(anchors + sample_uniform(remaining, min(budget, n_total) - len(anchors))))
        return selected
    if allocation == "event_equal" and weights is not None:
        raise ValueError("Explicit weights cannot be combined with event_equal allocation")
    if not nodes or budget <= 0:
        return []

    if len(nodes) > budget:


        return sample_uniform(sorted({i for n in nodes for i in n.frame_ids}), budget)

    if allocation == "event_equal":


        ordered = sorted(nodes, key=lambda n: -n.n_frames)
        picked = []
        for index, node in enumerate(ordered):
            quota = (budget - len(picked)) // (len(ordered) - index)
            picked.extend(sample_uniform(node.frame_ids, quota))
        picked = sorted(set(picked))
        return _dilate(picked, budget, n_total) if n_total and len(picked) < budget else picked

    if weights is None:
        w = [float(n.n_frames) for n in nodes]
    else:
        w = [max(float(x), 0.0) for x in weights][: len(nodes)]
        w += [0.0] * (len(nodes) - len(w))
    if sum(w) <= 0:
        w = [1.0] * len(nodes)

    total = sum(w)
    quotas = [max(1, int(round(budget * x / total))) for x in w]


    order = sorted(range(len(nodes)), key=lambda i: -w[i])
    while sum(quotas) > budget:
        for i in reversed(order):
            if sum(quotas) <= budget:
                break
            if quotas[i] > 1:
                quotas[i] -= 1
    while sum(quotas) < budget:
        for i in order:
            if sum(quotas) >= budget:
                break
            if quotas[i] < nodes[i].n_frames:
                quotas[i] += 1
        if all(quotas[i] >= nodes[i].n_frames for i in range(len(nodes))):
            break

    picked: List[int] = []
    for n, q in zip(nodes, quotas):
        picked.extend(sample_uniform(n.frame_ids, q))
    picked = sorted(set(picked))

    if n_total and len(picked) < budget:
        picked = _dilate(picked, budget, n_total)
    return picked


def _dilate(picked: Sequence[int], budget: int, n_total: int) -> List[int]:

    chosen = set(picked)
    radius = 1
    while len(chosen) < budget and radius < n_total:
        for i in sorted(picked):
            for j in (i - radius, i + radius):
                if 0 <= j < n_total and j not in chosen:
                    chosen.add(j)
                    if len(chosen) >= budget:
                        return sorted(chosen)
        radius += 1
    return sorted(chosen)


def _normalise(xs: Sequence[float]) -> List[float]:

    if not xs:
        return []
    lo, hi = min(xs), max(xs)
    if hi - lo < 1e-9:
        return [1.0] * len(xs)
    return [0.05 + 0.95 * (x - lo) / (hi - lo) for x in xs]


def selection_variants(
    nodes: Sequence[TreeNode], budget: int, n_total: int
) -> Dict[str, List[int]]:


    usable = [n for n in nodes if n.n_frames]
    rel = _normalise([n.relevance for n in usable])
    conf = _normalise([n.stop_conf for n in usable])

    return {
        "keynode": select_key_frames(nodes, budget, n_total),
        "keynode_equal": select_key_frames(
            nodes, budget, n_total, weights=[1.0] * len(usable)
        ),
        "keynode_rel": select_key_frames(nodes, budget, n_total, weights=rel),
        "keynode_conf": select_key_frames(nodes, budget, n_total, weights=conf),
        "keynode_rel_conf": select_key_frames(
            nodes, budget, n_total, weights=[r * c for r, c in zip(rel, conf)]
        ),
        "uniform": sample_uniform(list(range(n_total)), budget),
    }
