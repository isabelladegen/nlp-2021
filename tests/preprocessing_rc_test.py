from src.preprocessing_rc import *
from hamcrest import *


def test_load_rc_dataset():
    data = load_rc_dataset("train")
    assert_that(data.num_rows, equal_to(20431))
    assert_that(data.num_columns, equal_to(6))
