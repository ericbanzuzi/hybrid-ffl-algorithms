# Utils for shakespeare dataset
# Based on https://github.com/TalwalkarLab/leaf/blob/master/models/utils/language_utils.py

ALL_LETTERS = (
    "\n !\"&'(),-.0123456789:;>?ABCDEFGHIJKLMNOPQRSTUVWXYZ[]abcdefghijklmnopqrstuvwxyz}"
)
NUM_LETTERS = len(ALL_LETTERS)


def _one_hot(
    index: int,
    size: int,
) -> list:
    """Returns one-hot vector with given size and value 1 at given index."""
    vec = [0 for _ in range(size)]
    vec[int(index)] = 1
    return vec


def letter_to_vec(letter: str, ohe: bool = False) -> int:
    """Return one-hot representation of given letter"""
    index = ALL_LETTERS.find(letter)
    if ohe:
        return _one_hot(index, NUM_LETTERS)
    return index


def word_to_indices(
    word: str,
) -> list:
    """
    Returns a list of character indices based on input word
    """
    indices = []
    for c in word:
        indices.append(ALL_LETTERS.find(c))
    return indices
