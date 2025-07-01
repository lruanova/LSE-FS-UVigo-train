import torch
from fingerspelling_trainer.training.utils.alphabets import Alphabet
from fast_ctc_decode import beam_search  # type: ignore


def test_encode_decode_symmetry(alphabet: Alphabet):
    for tok_id, token in alphabet.NUM_TO_LETTER.items():
        roundtrip = alphabet.decode_label(torch.tensor(alphabet.encode_label(token)))
        assert roundtrip == token, f"Error with token '{token}' (id {tok_id})"

    samples = [
        "ACHO",
        "LLAVE",
        "CHALLENGE",
        "CACHARRO",
    ]
    for s in samples:
        ids = alphabet.encode_label(s)
        back = alphabet.decode_label(torch.tensor(ids))
        assert back == s, f"Incorrect: '{s}' -> {ids} -> '{back}'"

    print("✓ encode_label ↔ decode_label OK for multilabel tokens")


def test_beamsearch_path_symmetry(alphabet: Alphabet, beam_size: int = 3):
    vocab = [""] + [
        alphabet.NUM_TO_LETTER[i] for i in range(1, len(alphabet.LETTER_TO_NUM) + 1)
    ]

    targets = ["CHARRA"]
    batch = []
    T_max = max(len(t) for t in targets)

    for word in targets:

        ids = alphabet.encode_label(word)
        probs = torch.full((T_max * 2, len(vocab)), -float("inf"))

        t = 0
        for idx in ids:
            probs[t, 0] = 0.0
            t += 1
            probs[t, idx] = 0.0
            t += 1
        for i in range(t, T_max * 2):
            probs[i, 0] = 0.0

        probs = probs.softmax(-1)
        batch.append(probs)

    probs = torch.stack(batch, dim=1)

    decoded_strings = []
    for i in range(probs.shape[1]):
        seq, path = beam_search(
            probs[:, i, :].cpu().numpy(), vocab, beam_size=beam_size
        )
        decoded_strings.append(seq)
        recon = "".join(vocab[idx] for idx in path[::-1])

        print(f"\nWORD: {word}")
        print(f"Vocab: {vocab}")
        print(f"seq: '{seq}'")
        print(f"path: {path} (original order, length={len(path)})")
        print(f"path (reverse): {list(reversed(path))}")
        print("Reconstructed:", recon)

        assert recon == seq, f"beam path desalineado: '{seq}' vs '{recon}'"

    assert (
        decoded_strings == targets
    ), f"Error: Expected: {targets}\nResult: {decoded_strings}"
    print("beam_search path ↔ string OK for multilabel tokens")


if __name__ == "__main__":
    alf = Alphabet(
        alphabet={
            "A": 1,
            "B": 2,
            "C": 3,
            "CH": 4,
            "D": 5,
            "E": 6,
            "F": 7,
            "G": 8,
            "H": 9,
            "I": 10,
            "J": 11,
            "K": 12,
            "L": 13,
            "LL": 14,
            "M": 15,
            "N": 16,
            "Ñ": 17,
            "O": 18,
            "P": 19,
            "Q": 20,
            "R": 21,
            "RR": 22,
            "S": 23,
            "T": 24,
            "U": 25,
            "V": 26,
            "W": 27,
            "X": 28,
            "Y": 29,
            "Z": 30,
            "@": 31,
        }
    )

    test_encode_decode_symmetry(alf)
    test_beamsearch_path_symmetry(alf)
