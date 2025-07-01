import torch
import logging
from typing import List


class Alphabet:
    def __init__(self, alphabet: dict[str, int]):
        self.LETTER_TO_NUM = alphabet
        self.NUM_TO_LETTER = {v: k for k, v in self.LETTER_TO_NUM.items()}

    def get_letter_to_num_dict(self):
        return self.LETTER_TO_NUM

    def get_num_to_letter_dict(self):
        return self.NUM_TO_LETTER

    def __len__(self):
        return len(self.LETTER_TO_NUM)

    def encode_label(self, label: str) -> List[int]:
        label = label.upper()
        tokens = []
        i = 0

        while i < len(label):
            # check two character tokens
            if i + 1 < len(label):
                pair = label[i] + label[i + 1]
                if pair in self.LETTER_TO_NUM:
                    tokens.append(self.LETTER_TO_NUM[pair])
                    i += 2
                    continue
            # check one character tokens
            char = label[i]
            if char in self.LETTER_TO_NUM:
                tokens.append(self.LETTER_TO_NUM[char])
            else:
                logging.error(f"Character {char} not found in the alphabet")
            i += 1

        return tokens

    def decode_label(self, encoded_label: torch.Tensor) -> str:
        decoded_label = []
        for num in encoded_label:
            num = int(num.item())
            if num in self.NUM_TO_LETTER:
                decoded_label.append(self.NUM_TO_LETTER[num])
            else:
                logging.error(f"Number {num} not found in the alphabet")
        decoded_label = "".join(decoded_label)
        return decoded_label
