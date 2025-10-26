import pytest

from src.conversation.validation import _extract_budget


@pytest.mark.parametrize(
    "conv,expected",
    [
        (["I want a bike under 12k"], 12000.0),
        (["My budget is $8,500"], 8500.0),
        (["Budget: 12000 USD"], 12000.0),
        (["Looking for something up to 9k"], 9000.0),
        (["Budget <= 15k"], 15000.0),
        (["around 10k"], 10000.0),
        (["I have 12,000 dollars"], 12000.0),
        (["No budget specified"], None),
        (["budget 7000"], 7000.0),
        (["range 5k-8k"], 8000.0),
        (["approx. 4k"], 4000.0),
    ],
)
def test_extract_budget_various(conv, expected):
    b = _extract_budget(conv)
    assert b == expected
