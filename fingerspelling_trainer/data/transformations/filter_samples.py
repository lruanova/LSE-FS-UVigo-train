from keypoint_extraction_pipeline.schemas.annotation import AnnotationRecord


class FilterSamples:
    _MIN_FRAMES_THRESHOLD: int = 10

    def __call__(self, annotation: AnnotationRecord) -> AnnotationRecord | None:
        if not annotation.frames:
            return None

        if len(annotation.frames) <= self._MIN_FRAMES_THRESHOLD:
            print(
                f"🧹 Filtering annotation {annotation.metadata.video_path} because duration ({len(annotation.frames)}) is less or equal than {self._MIN_FRAMES_THRESHOLD} frames \n"
            )
            return None

        has_hand_keypoints = False
        for fr in annotation.frames:
            left_hand_has_keypoints = fr.left_hand and fr.left_hand.keypoints
            right_hand_has_keypoints = fr.right_hand and fr.right_hand.keypoints

            if left_hand_has_keypoints or right_hand_has_keypoints:
                has_hand_keypoints = True
                break

        if not has_hand_keypoints:
            print(
                f"🧹 Filtering annotation {annotation.metadata.video_path} because no keypoints found. \n"
            )
            return None

        return annotation
