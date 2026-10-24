import pytest

from cnn_lstm_bp.data import subject_split


def test_split():
    train, val, test = subject_split([f"s{i}" for i in range(10)], 38)
    assert train == ["s7", "s9", "s2", "s6", "s4", "s3", "s5", "s1"]
    assert val == ["s0"]
    assert test == ["s8"]
    wide = subject_split([f"s{i}" for i in range(100)], 38)
    assert [len(part) for part in wide] == [80, 10, 10]
    assert set(wide[0]).isdisjoint(wide[1]) and set(wide[0]).isdisjoint(wide[2])
    assert set(wide[1]).isdisjoint(wide[2])
    for ids in (["a", "b", "c", "d"], ["a", "b", "c", "d", "e"]):
        with pytest.raises(SystemExit):
            subject_split(ids, 38)
