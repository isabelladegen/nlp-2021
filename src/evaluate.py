from datasets import load_metric
import pandas as pd
from pandas import DataFrame

SCORE_EXACT = 'exact'
SCORE_F1 = 'f1'
SCORE_NUMBER_OF_QUESTIONS = 'total'


class PredictionsEvaluation:
    def __init__(self):
        self.likelihoods = {}
        self.predictions = []
        self.references = []

    def add(self, prediction_id, predicted_answer, gold_answer, likelihoods: list[float] = []):
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

        self.likelihoods[prediction_id] = likelihoods

    def squad2_score(self) -> [dict]:
        metric = load_metric("squad_v2")
        metric.add_batch(predictions=self.predictions, references=self.references)
        score = metric.compute()
        final_score = score
        return final_score

    def per_prediction_score(self) -> DataFrame:
        df = pd.DataFrame(columns=('id', 'f1', 'exact', 'likelihoods'))
        metric = load_metric("squad_v2")
        for index, prediction in enumerate(self.predictions):
            pred_id = prediction['id']
            metric.add(prediction=prediction, reference=self.__reference_for_id(pred_id))
            score = metric.compute()
            df.loc[index] = [pred_id, score[SCORE_F1], score[SCORE_EXACT], self.likelihoods[pred_id]]
        return df

    def __reference_for_id(self, ref_id: str):
        for reference in self.references:
            if reference['id'] == ref_id:
                return reference
