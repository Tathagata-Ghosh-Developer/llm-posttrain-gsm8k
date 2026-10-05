import math

from src.data import (clean_solution, extract_answer, gold_answer, is_correct, normalize_number, pass_at_k,
                      truncate_completion)


def test_gold_answer_after_hashes():
    assert gold_answer("She has 3 + 4 = <<3+4=7>>7 apples.\n#### 7") == "7"
    assert gold_answer("...\n#### 1,234") == "1234"


def test_clean_solution_removes_calculator_annotations():
    out = clean_solution("Half is 48/2 = <<48/2=24>>24.\n#### 24")
    assert "<<" not in out and out.endswith("#### 24")


def test_normalize_number():
    assert normalize_number("$18.00") == "18"
    assert normalize_number("0.50") == "0.5"
    assert normalize_number("-3") == "-3"
    assert normalize_number("abc") is None


def test_extract_answer_uses_last_number():
    assert extract_answer(" 2 + 3 = 5 apples, then 5 * 2 = 10.\n#### 10") == "10"
    assert extract_answer(" The total is 1,500 dollars.") == "1500"
    assert extract_answer(" no digits here") is None


def test_extract_answer_ignores_text_after_answer_line_and_next_question():
    text = " 4 * 3 = 12\n#### 12\nThat is 99 in total.\nQuestion: what is 7?"
    assert truncate_completion(text) == " 4 * 3 = 12\n#### 12"
    assert extract_answer(text) == "12"
    assert extract_answer(" So 8 eggs.\n\nQuestion: A farm has 50 cows") == "8"
    # the base model often writes "The answer is X." and then an invented "[Question] ..."
    assert extract_answer(" 16 - 3 = 13. The answer is 13.\n[Question]A farm has 50 cows") == "13"


def test_is_correct():
    assert is_correct(" ... #### 72", "72")
    assert not is_correct(" ... #### 71", "72")
    assert not is_correct(" nothing", "72")


def test_pass_at_k():
    assert pass_at_k(8, 0, 1) == 0.0
    assert pass_at_k(8, 8, 4) == 1.0
    assert math.isclose(pass_at_k(8, 1, 1), 1 / 8)
    assert pass_at_k(8, 1, 8) == 1.0
    # 1 - C(6,4)/C(8,4) = 1 - 15/70
    assert math.isclose(pass_at_k(8, 2, 4), 1 - 15 / 70)
