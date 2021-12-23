from src.preprocessing_rc import *
from hamcrest import *


def test_load_rc_dataset():
    data = load_rc_dataset("train")
    assert_that(data.num_rows, equal_to(20431))
    assert_that(data.num_columns, equal_to(6))


def test_returns_df_of_raw_question_and_gold_answers():
    data = RCData("train")

    raw_data = data.raw_questions_and_gold_answers()
    assert_that(raw_data.shape[0], equal_to(20431))