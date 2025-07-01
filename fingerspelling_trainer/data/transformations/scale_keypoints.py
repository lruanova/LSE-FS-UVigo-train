from __future__ import annotations
from pathlib import Path
from typing import Protocol, List

import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler
import os
import time
from torch.utils.data import DataLoader
from tqdm import tqdm


class _HasFrames(Protocol):
    class Frame(Protocol):
        left_hand: object | None
        right_hand: object | None
        left_hand_velocity: object | None
        right_hand_velocity: object | None

    frames: List[Frame]


class ScaleKeypoints:
    """
    Keeps two scalers (StandardScaler)
    - one for keypoints (x,y,z)
    - one for wrist_velocity (vx,vy,vz)

    Fits on training data with the dataloader using only valid keypoint data.
    Once scaler fitted, normalizes in-place keypoints and wrist-velocities on
    each AnnotationRecord
    """

    SLEEP_SECS = 0.3

    def __init__(self, scaler_dir: str | Path):
        self.dir = Path(scaler_dir)
        self.scaler_kp = StandardScaler()
        self.scaler_vel = StandardScaler()
        self._fitted = False

        kp_path = self.dir / "kp.pkl"
        vel_path = self.dir / "vel.pkl"
        if kp_path.exists() and vel_path.exists():
            self.scaler_kp = joblib.load(kp_path)  # type: ignore[arg-type]
            self.scaler_vel = joblib.load(vel_path)  # type: ignore[arg-type]
            self._fitted = True
            print(f"Scalers loaded from {self.dir} in __init__.")

    def __call__(self, annotation: _HasFrames):
        # If not fitted but file exists, load it
        if (
            (not self._fitted)
            and (self.dir / "kp.pkl").exists()
            and (self.dir / "vel.pkl").exists()
        ):
            time.sleep(self.SLEEP_SECS)
            self.scaler_kp = joblib.load(self.dir / "kp.pkl")  # type: ignore[arg-type]
            self.scaler_vel = joblib.load(self.dir / "vel.pkl")  # type: ignore[arg-type]
            self._fitted = True
            print(f"Scalers loaded from {self.dir} in __call__.")

        # If not fit, accumulate kps for fit latter
        if not self._fitted:
            print("ScaleKeypoints: Not fitted, returning annotation unmodified.")
            kp_coords: List[List[float]] = []
            for fr in annotation.frames:
                for side in ("left_hand", "right_hand"):
                    kp = getattr(fr, side)
                    if kp and kp.keypoints:
                        for p in kp.keypoints:
                            kp_coords.append([p.x or 0.0, p.y or 0.0, p.z or 0.0])
            if kp_coords:
                pass
            return annotation

        objs_kp: List[object] = []
        coords_kp: List[List[float]] = []
        objs_vel: List[object] = []
        coords_vel: List[List[float]] = []

        for fr in annotation.frames:
            for side in ("left_hand", "right_hand"):
                kp = getattr(fr, side)
                if kp and kp.keypoints:
                    for p in kp.keypoints:
                        objs_kp.append(p)
                        coords_kp.append([p.x or 0.0, p.y or 0.0, p.z or 0.0])

                # wrist-velocity
                vel = getattr(fr, f"{side}_velocity", None)
                if vel is not None:
                    x0, y0, z0 = vel.x or 0.0, vel.y or 0.0, vel.z or 0.0
                    if abs(x0) + abs(y0) + abs(z0) > 1e-6:
                        objs_vel.append(vel)
                        coords_vel.append([x0, y0, z0])

        # Transform kps
        if coords_kp:
            arr_kp = np.asarray(coords_kp, dtype=np.float32)
            arr_kp_norm = self.scaler_kp.transform(arr_kp)
            for obj, (x_, y_, z_) in zip(objs_kp, arr_kp_norm):
                obj.x, obj.y, obj.z = float(x_), float(y_), float(z_)  # type: ignore

        # transform wrist-vel
        if coords_vel:
            arr_vel = np.asarray(coords_vel, dtype=np.float32)
            arr_vel_norm = self.scaler_vel.transform(arr_vel)
            for obj, (x_, y_, z_) in zip(objs_vel, arr_vel_norm):
                obj.x, obj.y, obj.z = float(x_), float(y_), float(z_)  # type: ignore

        return annotation

    def fit_on_dataset(self, dataset, disable_augments: bool = True):
        if self._fitted:
            return

        rank = int(os.environ.get("RANK", "0"))
        if rank != 0:
            # other workers wait for file to exist
            while (
                not (self.dir / "kp.pkl").exists()
                or not (self.dir / "vel.pkl").exists()
            ):
                time.sleep(self.SLEEP_SECS)
            self.scaler_kp = joblib.load(self.dir / "kp.pkl")  # type: ignore
            self.scaler_vel = joblib.load(self.dir / "vel.pkl")  # type: ignore
            self._fitted = True
            return

        # main worker
        if disable_augments and hasattr(dataset, "transformations"):
            original_tfms = dataset.transformations.transforms
            dataset.transformations.transforms = [
                t for t in original_tfms if t.__class__.__name__ != "MirrorHands"
            ]

        loader = DataLoader(
            dataset,
            batch_size=128,
            shuffle=False,
            num_workers=16,
            collate_fn=lambda x: x,
        )

        self.scaler_kp = StandardScaler()
        self.scaler_vel = StandardScaler()
        PAD = getattr(dataset, "PAD_VALUE", 999.0)
        print(f"[rank0] Fitting scalers on {len(dataset)} samples…")

        for batch in tqdm(loader, total=len(loader)):
            kp_collect: List[np.ndarray] = []
            vel_collect: List[np.ndarray] = []

            for s in batch:
                lh = s["left_hand_keypoints"].reshape(-1, 3).cpu().numpy()
                lm = s["left_mask"].reshape(-1).cpu().numpy()
                rh = s["right_hand_keypoints"].reshape(-1, 3).cpu().numpy()
                rm = s["right_mask"].reshape(-1).cpu().numpy()

                # concat first, then filter by mask and padding
                all_kp = np.vstack([lh, rh])
                all_mask = np.concatenate([lm, rm]) == 1  # True for valid kps
                pts = all_kp[all_mask]
                # filter padding = [999,999,999]
                valid = np.abs(pts - PAD).sum(axis=1) > 1e-3
                pts = pts[valid]
                if pts.size:
                    kp_collect.append(pts)

                # add non null speeds
                vl = s["wrist_velocity_left"].cpu().numpy()  # [T,3]
                vr = s["wrist_velocity_right"].cpu().numpy()
                # filter (0,0,0)
                vl = vl[np.abs(vl).sum(axis=1) > 1e-6]
                vr = vr[np.abs(vr).sum(axis=1) > 1e-6]
                if vl.size:
                    vel_collect.append(vl)
                if vr.size:
                    vel_collect.append(vr)

            if kp_collect:
                all_kp_batch = np.vstack(kp_collect).astype(np.float32)
                self.scaler_kp.partial_fit(all_kp_batch)

            if vel_collect:
                all_vel_batch = np.vstack(vel_collect).astype(np.float32)
                self.scaler_vel.partial_fit(all_vel_batch)

        # save scalers
        self.dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.scaler_kp, self.dir / "kp.pkl")
        joblib.dump(self.scaler_vel, self.dir / "vel.pkl")
        self._fitted = True
        print(f"[rank0] ✅ Scalers guardados en {self.dir}")

        if disable_augments and hasattr(dataset, "transformations"):
            dataset.transformations.transforms = original_tfms
