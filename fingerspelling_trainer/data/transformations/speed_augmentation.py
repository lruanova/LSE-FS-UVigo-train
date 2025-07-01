# speed_augmentation.py  ────────────────────────────────────────────────
from __future__ import annotations
import random
import torch
from keypoint_extraction_pipeline.schemas.annotation import AnnotationRecord
from keypoint_extraction_pipeline.schemas.keypoints import Point3D


class SpeedJitter:

    def __init__(self, p: float = 0.9, min_scale: float = 0.5, max_scale: float = 2.3):
        if not 0.0 < p <= 1.0:
            raise ValueError("p ∈ (0,1]")
        if min_scale <= 0 or max_scale <= 0:
            raise ValueError("scale factors must be positive.")
        self.p, self.min_s, self.max_s = p, min_scale, max_scale

    def __call__(self, ann: AnnotationRecord) -> AnnotationRecord:
        if random.random() > self.p or len(ann.frames) < 3:
            return ann  # no jitter

        s = random.uniform(self.min_s, self.max_s)  # selected factor
        T0 = len(ann.frames)
        T1 = max(2, int(T0 / s))
        idx = torch.linspace(0, T0 - 1, T1)
        ann.frames = [ann.frames[int(i)] for i in idx]
        dt = 1.0 / s
        for fr in ann.frames:
            for v_attr in ("left_hand_velocity", "right_hand_velocity"):
                v: Point3D | None = getattr(fr, v_attr, None)
                if v is not None:
                    v.x = (v.x or 0.0) * dt
                    v.y = (v.y or 0.0) * dt
                    v.z = (v.z or 0.0) * dt
        return ann
