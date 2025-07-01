import re
import unicodedata

import torch
from fingerspelling_trainer.training.utils.alphabets import Alphabet
from keypoint_extraction_pipeline.schemas.annotation import AnnotationRecord


def _strip_accents_keep_ñ(text: str) -> str:
    chars = []
    for i, ch in enumerate(unicodedata.normalize("NFD", text)):
        if unicodedata.combining(ch):
            base = chars[-1] if chars else ""
            # keeps "Ñ"
            if ch == "\u0303" and base.upper() == "N":
                chars[-1] = "Ñ" if base.isupper() else "ñ"
            # remove other accents
            continue
        chars.append(ch)
    return unicodedata.normalize("NFC", "".join(chars))


class EncodeLabel:
    def __init__(
        self,
        alphabet: Alphabet,
        remove_non_alphabetic: bool = False,
        collapse_repeated: bool = False,
        include_spaces: bool = False,
        validate: bool = False,
    ):
        self.alphabet = alphabet
        self.remove_non_alphabetic = remove_non_alphabetic
        self.collapse_repeated = collapse_repeated
        self.include_spaces = include_spaces
        self.validate = validate

    def _format_label(self, label: str) -> str:
        # Remove "DT:" prefix
        label = label[3:] if label.startswith("DT:") else label
        label = label.upper()

        # remove accents but keeps "ñ"
        label = _strip_accents_keep_ñ(label)

        # filter allowed characters
        allowed = {"'", " ", "@", "_"}
        label = "".join(c for c in label if (c.isalpha() or c in allowed))

        # collapse repeated (double letters RR, LL) if set
        if self.collapse_repeated:
            label = re.sub(r"(.)\1+", r"\1", label)

        # include spaces if set, else removes them
        label = label.replace(" ", "_" if self.include_spaces else "")

        # removes other non alphabetic
        if self.remove_non_alphabetic:
            label = "".join(c for c in label if c.isalpha() or c == "@")

        return label

    def __call__(self, annotation: AnnotationRecord) -> AnnotationRecord:
        raw = annotation.metadata.label
        formatted = self._format_label(raw)
        tokens = self.alphabet.encode_label(formatted)

        if self.validate:
            decoded = self.alphabet.decode_label(torch.tensor(tokens))
            if decoded != formatted:
                raise ValueError(f"Bad label: '{formatted}' ≠ '{decoded}'")

        annotation.metadata.custom_properties["encoded_label"] = tokens
        return annotation
