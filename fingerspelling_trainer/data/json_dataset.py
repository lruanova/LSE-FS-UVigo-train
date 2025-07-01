from __future__ import annotations
from pathlib import Path

import torch
from torch.utils.data import Dataset
from typing import TypedDict, Any

from fingerspelling_trainer.data.compose_transforms import ComposeTransforms
from keypoint_extraction_pipeline.schemas.annotation import AnnotationRecord
from keypoint_extraction_pipeline.savers.json_saver import JSONSaver
from keypoint_extraction_pipeline.transformations.operators.base_operator import (
    BaseOperator,
)


class ProcessedSample(TypedDict):
    left_hand_keypoints: torch.Tensor  # [T, 21, 3]
    right_hand_keypoints: torch.Tensor
    left_mask: torch.Tensor  # [T, 21]  bool
    right_mask: torch.Tensor
    label: torch.Tensor  # concat labels [L]
    length: torch.Tensor  # nº of frames
    signing_hand: torch.Tensor  # 0=left, 1=right
    wrist_velocity_left: torch.Tensor  # [T,3]
    wrist_velocity_right: torch.Tensor  # [T,3]
    frames_per_char: float


class JSONDataset(Dataset):
    """Reads and apply transformations to the JSON files generated with the keypoint extractor tool."""

    ASSUMED_KEYPOINTS_PER_HAND = 21
    PAD_VALUE = 999.0

    def __init__(
        self,
        json_files_dir: str | Path,
        transformations: ComposeTransforms,
        require_label: bool = True,
    ):
        self.json_files_dir = Path(json_files_dir)
        self.transformations = transformations
        self.files = sorted(self.json_files_dir.glob("*.json"))
        self.require_label = require_label  # for inference
        if not self.files:
            raise FileNotFoundError(f"No .json found in {self.json_files_dir}")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> ProcessedSample:
        """
        Loads a JSON file and applies transforms.
        - If some transformation returns 'None', skip to next file until
        finding a valid one.

        """

        start_idx = idx
        while True:
            record: AnnotationRecord = JSONSaver.load_record(self.files[idx])
            # transforms pipeline
            record = self.transformations(annotation=record)  # type: ignore
            if record is not None:
                break

            # discard sample and skip to next one
            idx = (idx + 1) % len(self)
            if idx == start_idx:
                raise RuntimeError("Todas las muestras filtradas como vacías")

        # Tensor conversion
        def_frame = [
            [self.PAD_VALUE, self.PAD_VALUE, self.PAD_VALUE]
        ] * 21  # siempre 21
        def_mask = [0] * 21

        # buffers
        lh_data, rh_data = [], []
        lh_mask, rh_mask = [], []
        vel_l, vel_r = [], []

        for fr in record.frames:
            # ---------- LEFT hand points ----------
            if fr.left_hand and fr.left_hand.keypoints:
                kps_left = [
                    [p.x or 0, p.y or 0, p.z or 0] for p in fr.left_hand.keypoints
                ]
                mask_left = [
                    1 if BaseOperator.is_valid(p) else 0 for p in fr.left_hand.keypoints
                ]
            else:
                kps_left, mask_left = def_frame.copy(), def_mask.copy()

            # ---------- wrist velocity LEFT ----------
            if fr.left_hand_velocity:
                v = fr.left_hand_velocity
                vel_l.append([v.x or 0, v.y or 0, v.z or 0])
            else:
                vel_l.append([0.0, 0.0, 0.0])

            lh_data.append(kps_left)
            lh_mask.append(mask_left)

            # ---------- RIGHT hand points ----------
            if fr.right_hand and fr.right_hand.keypoints:
                kps_right = [
                    [p.x or 0, p.y or 0, p.z or 0] for p in fr.right_hand.keypoints
                ]
                mask_right = [
                    1 if BaseOperator.is_valid(p) else 0
                    for p in fr.right_hand.keypoints
                ]
            else:
                kps_right, mask_right = def_frame.copy(), def_mask.copy()

            # ---------- wrist velocity RIGHT ----------
            if fr.right_hand_velocity:
                v = fr.right_hand_velocity
                vel_r.append([v.x or 0, v.y or 0, v.z or 0])
            else:
                vel_r.append([0.0, 0.0, 0.0])

            rh_data.append(kps_right)
            rh_mask.append(mask_right)

        lh_tensor = torch.tensor(lh_data, dtype=torch.float32)
        rh_tensor = torch.tensor(rh_data, dtype=torch.float32)
        vel_l_t = torch.tensor(vel_l, dtype=torch.float32)  # [T,3]
        vel_r_t = torch.tensor(vel_r, dtype=torch.float32)

        lh_kp_mask = torch.tensor(lh_mask, dtype=torch.bool)
        rh_kp_mask = torch.tensor(rh_mask, dtype=torch.bool)

        length = torch.tensor(lh_tensor.shape[0], dtype=torch.long)

        encoded = record.metadata.custom_properties.get("encoded_label")
        if encoded is None and self.require_label:
            raise ValueError("Missing 'encoded_label' after transforms.")
        if encoded is None:
            encoded = []

        label = torch.tensor(encoded, dtype=torch.long)

        hand_str = record.metadata.handness
        if hand_str and hand_str.lower() == "left":
            signing_hand = torch.tensor(0, dtype=torch.long)
        elif hand_str and hand_str.lower() == "right":
            signing_hand = torch.tensor(1, dtype=torch.long)
        else:
            raise ValueError("Signing hand not found.")

        # >>> frames per char ---
        n_frames = int(length.item())
        n_chars = max(len(encoded), 1)  # Evita división por cero
        frames_per_char = float(n_frames) / n_chars

        return ProcessedSample(
            left_hand_keypoints=lh_tensor,
            right_hand_keypoints=rh_tensor,
            left_mask=lh_kp_mask,
            right_mask=rh_kp_mask,
            wrist_velocity_left=vel_l_t,
            wrist_velocity_right=vel_r_t,
            label=label,
            length=length,
            signing_hand=signing_hand,
            frames_per_char=frames_per_char,
        )

    # ========================================================
    #  Collate
    # ========================================================

    def _pad(self, seqs: list[torch.Tensor], T_max: int, value):
        if not seqs:
            return torch.empty(0)

        B = len(seqs)
        if seqs[0].dim() == 3:  # key-points  [T,21,3]
            _, N, C = seqs[0].shape
            out = torch.full(
                (B, T_max, N, C), value, dtype=seqs[0].dtype, device=seqs[0].device
            )
        else:  # máscaras ó velocidades  [T,21]  / [T,3]
            _, N = seqs[0].shape
            out = torch.full(
                (B, T_max, N), value, dtype=seqs[0].dtype, device=seqs[0].device
            )

        for i, s in enumerate(seqs):
            out[i, : s.shape[0]] = s
        return out

    def collate_batch(self, batch: list[ProcessedSample]) -> dict[str, Any]:
        lhs = [s["left_hand_keypoints"] for s in batch]
        rhs = [s["right_hand_keypoints"] for s in batch]
        lmsk = [s["left_mask"] for s in batch]
        rmsk = [s["right_mask"] for s in batch]

        vel_l = [s["wrist_velocity_left"] for s in batch]  # listas [T,3]
        vel_r = [s["wrist_velocity_right"] for s in batch]

        lengths = torch.stack([s["length"] for s in batch])  # [B]
        T_max = int(lengths.max())

        padded_lh = self._pad(lhs, T_max, value=self.PAD_VALUE)
        padded_rh = self._pad(rhs, T_max, value=self.PAD_VALUE)
        padded_lmsk = self._pad(lmsk, T_max, value=False)
        padded_rmsk = self._pad(rmsk, T_max, value=False)
        padded_vel_l = self._pad(vel_l, T_max, value=0.0)  # [B,T,3]
        padded_vel_r = self._pad(vel_r, T_max, value=0.0)

        # --- labels for CTC ---
        labels_list = [s["label"] for s in batch]
        labels_concat = torch.cat(labels_list)
        label_lengths = torch.tensor([t.numel() for t in labels_list])  # [B]

        # --- temporal mask of frames ---
        mask = (
            torch.arange(T_max, device=lengths.device)[None, :] < lengths[:, None]
        )  # [B,T]

        signing_hands = torch.stack([s["signing_hand"] for s in batch])  # [B]
        wrist_vel = torch.where(
            signing_hands[:, None, None] == 1, padded_vel_r, padded_vel_l  # 1 → right
        )

        frames_per_char = torch.tensor(
            [s["frames_per_char"] for s in batch], dtype=torch.float32
        )  # [B]

        return {
            "left": padded_lh,
            "right": padded_rh,
            "lengths": lengths,
            "labels": labels_concat,
            "label_lengths": label_lengths,
            "original_labels": labels_list,
            "mask": mask,
            "kp_mask_left": padded_lmsk,
            "kp_mask_right": padded_rmsk,
            "signing_hands": signing_hands,
            "wrist_vel": wrist_vel,
            "frames_per_char": frames_per_char,
        }
