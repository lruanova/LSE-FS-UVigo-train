from keypoint_extraction_pipeline.schemas.annotation import AnnotationRecord


class CanonicalizeHand:
    """Reflects left-hand keypoints (as well as wrist-velocity) in X-axis."""

    def __call__(self, record: AnnotationRecord) -> AnnotationRecord:
        if (record.metadata.handness or "").lower() != "left":
            return record

        for fr in record.frames:
            # keypoints
            if fr.left_hand and fr.left_hand.keypoints:
                for kp in fr.left_hand.keypoints:
                    if kp.x is not None:
                        kp.x = -kp.x
            # wrist velocity
            if fr.left_hand_velocity and fr.left_hand_velocity.x is not None:
                fr.left_hand_velocity.x = -fr.left_hand_velocity.x

        return record
