from hamcrest import *
from src.evaluate import PredictionsEvaluation
from utils.test_utils import *


def test_returns_full_score_for_predicting_gold_answer():
    answer_id = 'some id'
    gold = 'some answer prediction'
    reference_answer = ReferenceAnswerBuilder().with_id(answer_id).with_text(gold).build()

    predictions_evaluation = PredictionsEvaluation()
    predictions_evaluation.add(answer_id, gold, reference_answer)

    score = predictions_evaluation.squad2_score()

    assert_that(score['total'], len(predictions_evaluation.predictions))
    assert_that(score['f1'], equal_to(100.0))
    assert_that(score['exact'], equal_to(100.0))


def test_returns_zero_for_not_predicting_a_single_word():
    # build some fake prediction
    answer_id1 = 'id1'
    prediction = 'some answer prediction'
    reference_answer1 = ReferenceAnswerBuilder().with_id(answer_id1).with_text('totally different').build()

    predictions_evaluation = PredictionsEvaluation()
    predictions_evaluation.add(answer_id1, prediction, reference_answer1)

    score = predictions_evaluation.squad2_score()

    assert_that(score['total'], len(predictions_evaluation.predictions))
    assert_that(score['f1'], equal_to(0.0))
    assert_that(score['exact'], equal_to(0.0))


def test_returns_50_for_predicting_half_the_text():
    # build some fake prediction
    answer_id1 = 'id1'
    prediction = 'some answer'
    reference_answer1 = ReferenceAnswerBuilder().with_id(answer_id1).with_text('some other').build()

    predictions_evaluation = PredictionsEvaluation()
    predictions_evaluation.add(answer_id1, prediction, reference_answer1)

    score = predictions_evaluation.squad2_score()

    assert_that(score['total'], len(predictions_evaluation.predictions))
    assert_that(score['f1'], equal_to(50.0))
    assert_that(score['exact'], equal_to(0.0))


def test_returns_50_for_predicting_half_the_gold_answers():
    # build some fake predictions
    answer_id1 = 'id1'
    gold1 = 'some answer prediction'
    reference_answer1 = ReferenceAnswerBuilder().with_id(answer_id1).with_text(gold1).build()

    answer_id2 = 'id2'
    gold2 = 'another'
    reference_answer2 = ReferenceAnswerBuilder().with_id(answer_id2).with_text(gold2).build()

    answer_id3 = 'false'
    gold3 = 'some other lenghty blad di bladeruoisuie text what ever'
    reference_answer3 = ReferenceAnswerBuilder().with_id(answer_id3).with_text("totally different").build()

    answer_id4 = 'id4'
    gold4 = 'some other'
    reference_answer4 = ReferenceAnswerBuilder().with_id(answer_id4).with_text("different").build()

    predictions_evaluation = PredictionsEvaluation()
    predictions_evaluation.add(answer_id1, gold1, reference_answer1)
    predictions_evaluation.add(answer_id2, gold2, reference_answer2)
    predictions_evaluation.add(answer_id3, gold3, reference_answer3)
    predictions_evaluation.add(answer_id4, gold4, reference_answer4)

    score = predictions_evaluation.squad2_score()

    assert_that(score['total'], len(predictions_evaluation.predictions))
    assert_that(score['f1'], equal_to(50.0))
    assert_that(score['exact'], equal_to(50.0))


def test_returns_dataframe_of_per_question_scores():
    # build some fake predictions
    answer_id1 = 'id1'
    gold1 = 'some answer prediction'
    reference_answer1 = ReferenceAnswerBuilder().with_id(answer_id1).with_text(gold1).build()

    answer_id2 = 'id2'
    gold2 = 'another'
    reference_answer2 = ReferenceAnswerBuilder().with_id(answer_id2).with_text(gold2).build()

    answer_id3 = 'false'
    gold3 = 'some other lenghty blad di bladeruoisuie text what ever'
    reference_answer3 = ReferenceAnswerBuilder().with_id(answer_id3).with_text("totally different").build()

    answer_id4 = 'id4'
    gold4 = 'some other'
    reference_answer4 = ReferenceAnswerBuilder().with_id(answer_id4).with_text("different").build()

    predictions_evaluation = PredictionsEvaluation()
    predictions_evaluation.add(answer_id1, gold1, reference_answer1, [1, 2])
    predictions_evaluation.add(answer_id2, gold2, reference_answer2)
    predictions_evaluation.add(answer_id3, gold3, reference_answer3, [1])
    predictions_evaluation.add(answer_id4, gold4, reference_answer4, [1, 2, 3, 4])

    df = predictions_evaluation.per_prediction_score()
    assert_that(df.shape, equal_to((4, 4)))

    df.set_index("id", inplace=True)
    assert_that(df.at[answer_id1, 'f1'], equal_to(100.0))
    assert_that(df.at[answer_id2, 'f1'], equal_to(100.0))
    assert_that(df.at[answer_id3, 'f1'], equal_to(0.0))
    assert_that(df.at[answer_id4, 'f1'], equal_to(0.0))

    assert_that(df.at[answer_id1, 'exact'], equal_to(100.0))
    assert_that(df.at[answer_id2, 'exact'], equal_to(100.0))
    assert_that(df.at[answer_id3, 'exact'], equal_to(0.0))
    assert_that(df.at[answer_id4, 'exact'], equal_to(0.0))

    assert_that(df.at[answer_id1, 'likelihoods'], equal_to([1, 2]))
    assert_that(df.at[answer_id2, 'likelihoods'], equal_to([]))
    assert_that(df.at[answer_id3, 'likelihoods'], equal_to([1]))
    assert_that(df.at[answer_id4, 'likelihoods'], equal_to([1, 2, 3, 4]))




def test_returns_full_score_for_predictdfing_all_gold_answers():
    # build some fake predictions
    answer_id1 = 'id1'
    gold1 = 'some answer prediction'
    reference_answer1 = ReferenceAnswerBuilder().with_id(answer_id1).with_text(gold1).build()

    answer_id2 = 'id2'
    gold2 = 'another'
    reference_answer2 = ReferenceAnswerBuilder().with_id(answer_id2).with_text(gold2).build()

    answer_id3 = 'id3'
    gold3 = 'some other lenghty blad di bladeruoisuie text what ever'
    reference_answer3 = ReferenceAnswerBuilder().with_id(answer_id3).with_text(gold3).build()

    predictions_evaluation = PredictionsEvaluation()
    predictions_evaluation.add(answer_id1, gold1, reference_answer1)
    predictions_evaluation.add(answer_id2, gold2, reference_answer2)
    predictions_evaluation.add(answer_id3, gold3, reference_answer3)

    score = predictions_evaluation.squad2_score()

    assert_that(score['total'], len(predictions_evaluation.predictions))
    assert_that(score['f1'], equal_to(100.0))
    assert_that(score['exact'], equal_to(100.0))
