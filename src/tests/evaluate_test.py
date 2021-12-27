from hamcrest import *
from src.evaluate import PredictionsEvaluation


class ReferenceAnswerBuilder:
    id: str
    answer_text: str
    answer_start: int

    def __init__(self):
        self.answer_start = 12345
        self.answer_text = "Answer text"
        self.id = 'some id'

    def build(self):
        squad2_metric_answer = {
            'id': self.id,
            'answers': {
                'text': [
                    self.answer_text
                ],
                'answer_start': [
                    self.answer_start
                ]
            }
        }
        return squad2_metric_answer

    def with_id(self, answer_id):
        self.id = answer_id
        return self

    def with_text(self, answer_text):
        self.answer_text = answer_text
        return self


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


def test_returns_full_score_for_predicting_all_gold_answers():
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


def test_returns_score():
    print("please implement")
    # 1. Train models
    #    - pre process document data -> Grounding Docs
    #    - tune model parameters -> Batch Trainer
    #    DATA: Documents dataset
    #
    # 2. Predict rc question
    #    - pre process rc question -> RCData
    #    - predict answers and create predictions and references -> BatchTrainer
    #    DATA: RC dataset
    #
    # 3. Evaluate results
    #    - Squad 2 metrics -> Evaluate

    # trainer = BatchTrainer(bla)
    # trainer.pre
    # rc_dataset =
    # Prediction()
