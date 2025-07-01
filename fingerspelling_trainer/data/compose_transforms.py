from typing import Any, Callable, List

Transform = Callable[[Any], Any]


class ComposeTransforms:
    def __init__(self, transforms: List[Transform]):
        self.transforms = transforms

    def __call__(self, annotation):
        for t in self.transforms:
            # if a previous transformation returned None, stop
            if annotation is None:
                return None

            annotation = t(annotation)

        return annotation
