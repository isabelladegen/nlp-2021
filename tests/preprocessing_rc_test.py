from hamcrest import *
from src.preprocessing_rc import *
from src.preprocessing_documents import preprocess_doc


def test_load_rc_dataset():
    data = load_rc_dataset("train")
    assert_that(data.num_rows, equal_to(20431))
    assert_that(data.num_columns, equal_to(6))


def test_by_default_does_a_simple_doc_to_vec_pre_process():
    user_question = "some user question that needs pre processing 5?"
    expected = preprocess_doc(user_question)

    pre_processed_question = preprocess_question(user_question)

    assert_that(pre_processed_question, expected)


def test_removes_dialogue_history_from_user_question():
    question_without_history = "Thanks, and in case I forget to bring all of the documentation needed " \
                               "to the DMV office, what can I do?"
    rc_user_question_string = "user:" \
                              + question_without_history + \
                              " agent:Yes, you can sign up for" \
                              " MyDMV for all the online transactions needed. user:Can I do my DMV " \
                              "transactions online? agent:hi, you have to report any change of address to" \
                              " DMV within 10 days after moving. You should do this both for the address " \
                              "associated with your license and all the addresses associated with all your " \
                              "vehicles. user:Hello, I forgot o update my address, can you help me with that?"
    expected = preprocess_doc(question_without_history)

    config = {'pre_process_rc_question': QuestionPreProcessing.user_question_only.value}
    pre_processed_question = preprocess_question(rc_user_question_string, config)

    assert_that(pre_processed_question, equal_to(expected))


def test_removes_dialogue_history_works_if_theres_no_history():
    question_without_history = "Thanks, and in case I forget to bring all of the documentation needed " \
                               "to the DMV office, what can I do?"
    rc_user_question_string = "user:" \
                              + question_without_history
    expected = preprocess_doc(question_without_history)

    config = {'pre_process_rc_question': QuestionPreProcessing.user_question_only.value}
    pre_processed_question = preprocess_question(rc_user_question_string, config)

    assert_that(pre_processed_question, equal_to(expected))
