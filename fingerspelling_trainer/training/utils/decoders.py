import torch

# import editdistance # https://pypi.org/project/editdistance/
from fast_ctc_decode import beam_search  # type: ignore
from fingerspelling_trainer.training.utils.alphabets import Alphabet
from itertools import groupby


def batched_beam_search_decoder(
    alphabet: Alphabet,
    targets_for_decoding: list[torch.Tensor],
    masked_log_probs,
    input_lengths,
    beam_size,
    verbose=False,
):
    """
    Perform beam search decoding on a batch of masked log probabilities.

    Args:
        text_transform: TextTransform object
        targets: List of target sequences
        masked_log_probs: Tensor of shape (seq_len, batch_size, num_classes)
        input_lengths: Tensor of shape (batch_size,)
    """
    probs = torch.exp(masked_log_probs)  # Convert log probabilities to probabilities

    decoded_preds = []
    decoded_targets = [alphabet.decode_label(t) for t in targets_for_decoding]

    # adding blank token to the alphabet
    vocab = [""] + [
        alphabet.NUM_TO_LETTER[i] for i in range(1, len(alphabet.LETTER_TO_NUM) + 1)
    ]

    for i in range(probs.shape[1]):  # iterating over batch size
        input_len = input_lengths[i].item()  # Get the actual length of the sequence
        trimmed_probs = (
            probs[:input_len, i, :].cpu().detach().numpy()
        )  # Trim the probs to the actual length

        sequence, path = beam_search(trimmed_probs, vocab, beam_size=beam_size)
        decoded_preds.append(sequence)

    if verbose:
        print("-----\n")
        print(f"Decoded: {decoded_preds}")
        print(f"Targets: {decoded_targets}")
        print("-----\n")

    return decoded_preds, decoded_targets


def greedy_decoder(
    alphabet: Alphabet,
    targets_for_decoding: list[torch.Tensor],
    masked_log_probs: torch.Tensor,
    input_lengths: torch.Tensor,
    verbose: bool = False,
) -> tuple[list[str], list[str]]:
    decoded_targets = [alphabet.decode_label(t) for t in targets_for_decoding]
    decoded_preds = []

    # frame‐wise argmax
    argmaxes = torch.argmax(masked_log_probs.exp(), dim=-1).permute(1, 0)  # (B, T)

    for i, args in enumerate(argmaxes):
        raw = args[: input_lengths[i]].cpu().tolist()
        # colapsar blanks + repeticiones
        out = []
        prev = None
        for tok in raw:
            if tok != 0 and tok != prev:
                out.append(tok)
            prev = tok
        decoded_preds.append(alphabet.decode_label(torch.tensor(out)))

    if verbose:
        print("-----\n")
        print(f"Decoded: {decoded_preds}")
        print(f"Targets: {decoded_targets}")
        print("-----\n")

    return decoded_preds, decoded_targets


def greedy_pause_decoder(
    alphabet: Alphabet,
    targets_for_decoding: list[torch.Tensor],
    masked_log_probs: torch.Tensor,
    input_lengths: torch.Tensor,
    verbose: bool = False,
    min_pause: int = 20,
) -> tuple[list[str], list[str]]:
    """
    Test with Greedy CTC decoding + insertion of '_' tokens when there are min_pause blanks.
      - if token == blank y run_length >= min_pause -> '_'
      - if token == blank y run_length < min_pause -> skipped
      - if token != blank -> token

    """
    decoded_targets = [alphabet.decode_label(t) for t in targets_for_decoding]
    decoded_preds = []
    blank_id = 0
    space_id = alphabet.LETTER_TO_NUM.get("_")
    if space_id is None:
        raise ValueError("Alphabet must contain '_' for decoding.")

    for i in range(masked_log_probs.shape[1]):
        # raw argmax per frame
        raw = (
            torch.argmax(masked_log_probs[:, i, :].exp(), dim=1)[: input_lengths[i]]
            .cpu()
            .tolist()
        )
        out_ids = []

        for token, group in groupby(raw):
            run = len(list(group))
            if token == blank_id:
                if run >= min_pause:
                    # long pause -> '_'
                    if not out_ids or out_ids[-1] != space_id:
                        out_ids.append(space_id)
                # short pause -> skip blanks
            else:
                out_ids.append(token)
        pred = alphabet.decode_label(torch.tensor(out_ids))
        decoded_preds.append(pred)
        if verbose:
            print("-----\n")
            print(f"Decoded: {decoded_preds}")
            print(f"Targets: {decoded_targets}")
            print("-----\n")

    return decoded_preds, decoded_targets
