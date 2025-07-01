from keypoint_extraction_pipeline.schemas.annotation import AnnotationRecord


class RemoveEmptySamples:
    def __call__(self, annotation: AnnotationRecord) -> AnnotationRecord | None:
        for fr in annotation.frames:
            if (fr.left_hand and fr.left_hand.keypoints) or (
                fr.right_hand and fr.right_hand.keypoints
            ):
                return annotation
        return None
