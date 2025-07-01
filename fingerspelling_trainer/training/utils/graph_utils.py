from __future__ import annotations
from typing import Dict, List, Tuple, Sequence
from enum import Enum

import torch
from torch_geometric.utils import to_undirected
from mediapipe.python.solutions.hands_connections import HAND_CONNECTIONS
from mediapipe.python.solutions.pose_connections import POSE_CONNECTIONS


class LandmarkType(str, Enum):
    LEFT_HAND = "left_hand"
    RIGHT_HAND = "right_hand"
    POSE = "pose"


_KEYPOINTS = {
    LandmarkType.LEFT_HAND: 21,
    LandmarkType.RIGHT_HAND: 21,
    LandmarkType.POSE: 33,
}


class EdgeIndexBuilder:
    """
    Creates an edge_index on PyG format [2,E] for each landmark subset.
    Supported strategies:
        - 'mediapipe' : using MediaPipe official connections.
        - 'functional' : using dense connections between groups (tips, knuckles...)
        - 'from_config' : uses edges from config file (hydra)
    """

    def __init__(
        self,
        strategy: str,
        selected_landmarks: Sequence[LandmarkType],
        undirected: bool = True,
        edges_from_cfg: Dict[str, List[Tuple[int, int]]] | None = None,
    ):
        self.strategy = strategy.lower()
        self.undirected = undirected
        self.edges_from_cfg = edges_from_cfg or {}

        if self.strategy not in {"mediapipe", "functional", "from_config"}:
            raise ValueError(f"Strategy: {self.strategy} not supported")

        # convert strings to landmark type
        self.selected_landmarks: list[LandmarkType] = []
        for lm in selected_landmarks:
            if isinstance(lm, str):
                try:
                    self.selected_landmarks.append(LandmarkType[lm])
                except KeyError:
                    raise ValueError(f"Unknown landmark type '{lm}'")
            else:
                self.selected_landmarks.append(lm)

    def build(self) -> Dict[LandmarkType, torch.Tensor]:
        if self.strategy == "mediapipe":
            return self._mediapipe()
        if self.strategy == "functional":
            return self._functional()
        return self._from_config()

    def add_velocity_node(
        self,
        edge_index: torch.Tensor,
        strategy: str = "wrist",  # 'wrist' | 'full' | 'fingers'
        wrist_idx: int = 0,
        vel_idx: int = 21,
    ):
        if strategy == "wrist":
            extra = torch.tensor(
                [[wrist_idx, vel_idx], [vel_idx, wrist_idx]], dtype=torch.long
            )
        elif strategy == "full":
            nodes = torch.arange(0, vel_idx, dtype=torch.long)
            extra = torch.stack(
                [
                    torch.cat([nodes, torch.full_like(nodes, vel_idx)]),
                    torch.cat([torch.full_like(nodes, vel_idx), nodes]),
                ]
            )
        elif strategy == "fingers":
            finger_tips = torch.tensor([4, 8, 12, 16, 20, wrist_idx])
            extra = torch.stack(
                [
                    torch.cat([finger_tips, torch.full_like(finger_tips, vel_idx)]),
                    torch.cat([torch.full_like(finger_tips, vel_idx), finger_tips]),
                ]
            )
        else:
            raise ValueError("vel_conn must be wrist|full|fingers")
        return torch.cat([edge_index, extra], dim=1)

    def _mediapipe(self):
        return {
            lm: self._to_tensor(
                list(HAND_CONNECTIONS if "hand" in lm else POSE_CONNECTIONS)
            )
            for lm in self.selected_landmarks
        }

    def _functional(self):
        def dense(nodes: List[int]) -> List[Tuple[int, int]]:
            return [(i, j) for k, i in enumerate(nodes) for j in nodes[k + 1 :]]

        edges_per_lm: Dict[LandmarkType, List[Tuple[int, int]]] = {}
        for lm in self.selected_landmarks:
            if "hand" not in lm:
                raise ValueError("Functional only defined for hands.")
            edges = list(HAND_CONNECTIONS)

            # grupos
            tips = [4, 8, 12, 16, 20]
            middles = [3, 7, 11, 15, 19]
            bases = [2, 6, 10, 14, 18]
            knuckles = [1, 5, 9, 13, 17]

            for group in (tips, middles, bases, knuckles):
                edges += dense(group)

            edges_per_lm[lm] = edges

        return {lm: self._to_tensor(edges) for lm, edges in edges_per_lm.items()}

    def _from_config(self):
        if not self.edges_from_cfg:
            raise ValueError(
                "edges_from_cfg cannot be empty with strategy='from_config'."
            )

        tensors = {}
        for lm in self.selected_landmarks:
            raw_edges = self.edges_from_cfg.get(lm.value)
            if raw_edges is None:
                raise KeyError(f"No edges found for {lm} on edges_from_cfg.")
            tensors[lm] = self._to_tensor(raw_edges)
        return tensors

    def _to_tensor(self, edges: Sequence[Tuple[int, int]]) -> torch.Tensor:
        edge_tensor = torch.tensor(edges, dtype=torch.long).t().contiguous()
        return to_undirected(edge_tensor) if self.undirected else edge_tensor
