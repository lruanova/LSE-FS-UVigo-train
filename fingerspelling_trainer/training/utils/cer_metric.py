import numpy as np


def _levenshtein_distance(ref, hyp):
    m, n = len(ref), len(hyp)

    if ref == hyp:
        return 0, 0, 0, 0
    if m == 0:
        return n, 0, n, 0
    if n == 0:
        return m, 0, 0, m

    distance = np.zeros((2, n + 1), dtype=np.int32)

    for j in range(n + 1):
        distance[0][j] = j

    substitutions, insertions, deletions = 0, 0, 0

    for i in range(1, m + 1):
        current_row = i % 2
        previous_row = 1 - current_row
        distance[current_row][0] = i

        for j in range(1, n + 1):
            if ref[i - 1] == hyp[j - 1]:
                cost = 0
            else:
                cost = 1

            options = (
                distance[previous_row][j - 1] + cost,  # Substitution
                distance[current_row][j - 1] + 1,  # Insertion
                distance[previous_row][j] + 1,
            )  # Deletion

            min_cost = min(options)
            distance[current_row][j] = min_cost

            # Count the operation based on which option was chosen
            if min_cost == distance[previous_row][j - 1] + cost:
                if cost == 1:
                    substitutions += 1
            elif min_cost == distance[current_row][j - 1] + 1:
                insertions += 1
            elif min_cost == distance[previous_row][j] + 1:
                deletions += 1

    return distance[m % 2][n], substitutions, insertions, deletions


def char_errors(reference, hypothesis, ignore_case=False, remove_space=False):
    """Compute the levenshtein distance between reference sequence and
    hypothesis sequence in char-level.
    :param reference: The reference sentence.
    :type reference: basestring
    :param hypothesis: The hypothesis sentence.
    :type hypothesis: basestring
    :param ignore_case: Whether case-sensitive or not.
    :type ignore_case: bool
    :param remove_space: Whether remove internal space characters
    :type remove_space: bool
    :return: Levenshtein distance and length of reference sentence.
    :rtype: list
    """
    if ignore_case == True:
        reference = reference.lower()
        hypothesis = hypothesis.lower()

    join_char = " "
    if remove_space == True:
        join_char = ""

    reference = join_char.join(filter(None, reference))
    hypothesis = join_char.join(filter(None, hypothesis))

    edit_distance, subs, ins, dels = _levenshtein_distance(reference, hypothesis)
    return float(edit_distance), len(reference), subs, ins, dels


def cer(
    reference, hypothesis, ignore_case=False, remove_space=False, return_counts=False
):
    """Calculate charactor error rate (CER). CER compares reference text and
    hypothesis text in char-level. CER is defined as:
    .. math::
        CER = (Sc + Dc + Ic) / Nc
    where
    .. code-block:: text
        Sc is the number of characters substituted,
        Dc is the number of characters deleted,
        Ic is the number of characters inserted
        Nc is the number of characters in the reference
    We can use levenshtein distance to calculate CER. Chinese input should be
    encoded to unicode. Please draw an attention that the leading and tailing
    space characters will be truncated and multiple consecutive space
    characters in a sentence will be replaced by one space character.
    :param reference: The reference sentence.
    :type reference: basestring
    :param hypothesis: The hypothesis sentence.
    :type hypothesis: basestring
    :param ignore_case: Whether case-sensitive or not.
    :type ignore_case: bool
    :param remove_space: Whether remove internal space characters
    :type remove_space: bool
    :return: Character error rate.
    :rtype: float
    :raises ValueError: If the reference length is zero.
    """
    edit_distance, ref_len, subs, ins, dels = char_errors(
        reference, hypothesis, ignore_case, remove_space
    )

    if ref_len == 0:
        raise ValueError("Length of reference should be greater than 0.")

    cer = float(edit_distance) / ref_len
    if return_counts:
        return cer, ref_len, subs, ins, dels
    return cer
