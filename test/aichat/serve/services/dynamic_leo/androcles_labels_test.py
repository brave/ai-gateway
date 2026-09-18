import pytest

from aichat.serve.services.dynamic_leo.androcles_labels import androcles_label_index


@pytest.mark.parametrize(
    "name,expect",
    [
        ("Brave", 0),
        ("brave", 0),
        ("Coding", 3),
        ("Math / Calculations", 9),
        ("Summarisation", 15),
        ("bogus", None),
        ("", None),
    ],
)
def test_androcles_label_index(name, expect):
    assert androcles_label_index(name) == expect
