import pytest

from ..preprocessing import *

@pytest.mark.parametrize("test_input, expected", [
    ("1+2", 3),  # min = 1, max = 3
])
def test_adder(test_input, expected):
    assert eval(test_input) == expected