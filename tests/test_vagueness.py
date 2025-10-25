from src.conversation.history import is_vague_input


def test_greeting_hi():
    assert is_vague_input("hi") is True


def test_greeting_how_are_you():
    assert is_vague_input("how are you?") is True


def test_short_pleasantry_whats_up():
    assert is_vague_input("what's up") is True


def test_substantive_short_two_tokens():
    # two informative tokens should be considered non-vague
    assert is_vague_input("long-travel suspension") is False


def test_budget_mention_is_not_vague():
    assert is_vague_input("budget $10k") is False


def test_question_with_informative_tokens():
    assert is_vague_input("looking for touring bike with good suspension") is False
