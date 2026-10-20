import itertools
import random

from utils.myers_diff import myers_diff


def _reconstruct(a, b, ops):
    out = []
    pos_i = 0
    pos_j = 0
    for tag, i1, i2, j1, j2 in ops:
        assert i1 == pos_i and j1 == pos_j
        if tag == "equal":
            assert a[i1:i2] == b[j1:j2]
            out.extend(a[i1:i2])
        elif tag == "delete":
            assert j1 == j2
        elif tag == "insert":
            assert i1 == i2
            out.extend(b[j1:j2])
        elif tag == "replace":
            out.extend(b[j1:j2])
        else:
            raise AssertionError(tag)
        pos_i, pos_j = i2, j2
    assert pos_i == len(a) and pos_j == len(b)
    return out


def test_myers_covers_empty_identical_and_edits():
    cases = [
        ([], []),
        (list("hello"), list("hello")),
        ([], list("abc")),
        (list("abc"), []),
        (list("abc"), list("axc")),
        (list("abc"), list("abcabc")),
        (list(range(10)), list(range(5, 15))),
    ]
    for a, b in cases:
        ops = myers_diff(a, b)
        assert ops is not None
        assert _reconstruct(a, b, ops) == b


def test_myers_random_sequences_reconstruct_the_target():
    rng = random.Random(0)
    cases = []
    for n, m in itertools.product(range(4), repeat=2):
        for a in itertools.product("ab", repeat=n):
            for b in itertools.product("ab", repeat=m):
                cases.append((list(a), list(b)))
    for _ in range(80):
        alphabet = rng.choice(("ab", "abc", "abcd"))
        a = [rng.choice(alphabet) for _ in range(rng.randint(0, 30))]
        b = [rng.choice(alphabet) for _ in range(rng.randint(0, 30))]
        cases.append((a, b))

    for a, b in cases:
        ops = myers_diff(a, b)
        assert ops is not None
        assert _reconstruct(a, b, ops) == b
