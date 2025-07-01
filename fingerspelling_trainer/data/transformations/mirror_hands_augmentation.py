import copy
import random
from keypoint_extraction_pipeline.schemas.annotation import AnnotationRecord


class MirrorHands:
    """
    Transform with a probability p, mirroring L/R hands,
    interchanging keypoints (x = -x), velocities and handness.
    """

    def __init__(self, p: float = 0.5):
        assert 0.0 <= p <= 1.0
        self.p = p

    def __call__(self, annotation: AnnotationRecord) -> AnnotationRecord:
        if random.random() > self.p:
            return annotation
        aug = copy.deepcopy(annotation)

        for fr in aug.frames:
            # Swap keypoints
            original_left_hand_data = fr.left_hand
            original_right_hand_data = fr.right_hand

            fr.left_hand = original_right_hand_data
            fr.right_hand = original_left_hand_data

            # Swap velocities
            original_left_vel = fr.left_hand_velocity
            original_right_vel = fr.right_hand_velocity
            fr.left_hand_velocity = original_right_vel
            fr.right_hand_velocity = original_left_vel

            # Mirror x for keypoints
            if fr.left_hand and fr.left_hand.keypoints:
                for kp in fr.left_hand.keypoints:
                    if kp.x is not None:
                        kp.x = -kp.x

            if fr.right_hand and fr.right_hand.keypoints:
                for kp in fr.right_hand.keypoints:
                    if kp.x is not None:
                        kp.x = -kp.x

            # Mirror x for velocities
            for v in (fr.left_hand_velocity, fr.right_hand_velocity):
                if v is not None and v.x is not None:
                    v.x = -v.x

        # Swap handness in metadata
        h = aug.metadata.handness
        if h:
            h_lower = h.lower()
            if h_lower == "right":
                aug.metadata.handness = "left"
            elif h_lower == "left":
                aug.metadata.handness = "right"

        return aug
