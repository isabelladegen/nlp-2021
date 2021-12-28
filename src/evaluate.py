from datasets import load_metric

SCORE_EXACT = 'exact'
SCORE_F1 = 'f1'
SCORE_NUMBER_OF_QUESTIONS = 'total'


class PredictionsEvaluation:
    def __init__(self):
        self.predictions = []
        self.references = []

    def add(self, prediction_id, predicted_answer, gold_answer):
        self.predictions.append(
            {'id': prediction_id,
             'prediction_text':
                 predicted_answer,
             'no_answer_probability': 0.0
             }
        )

        self.references.append(
            {
                "id": prediction_id,
                "answers": gold_answer,
            }
        )

    def squad2_score(self) -> [dict]:
        metric = load_metric("squad_v2")
        metric.add_batch(predictions=self.predictions, references=self.references)

        final_score = metric.compute()
        return final_score
