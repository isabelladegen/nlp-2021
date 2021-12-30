from datasets import load_metric

SCORE_EXACT = 'exact'
SCORE_F1 = 'f1'
SCORE_NUMBER_OF_QUESTIONS = 'total'


class PredictionsEvaluation:
    def __init__(self):
        self.likelihoods = []
        self.predictions = []
        self.references = []

    def add(self, prediction_id, predicted_answer, gold_answer, likelihoods: list[float] = None):
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

        if likelihoods:
            for likelihood in likelihoods:
                self.likelihoods.append(
                    {
                        'id': prediction_id,
                        'likelihood': likelihood
                    }
                )

    def squad2_score(self, wandb=None) -> [dict]:
        if wandb:
            self.__per_prediction_score(wandb)
        final_score = self.__overall_score()
        return final_score

    def __overall_score(self):
        metric = load_metric("squad_v2")
        metric.add_batch(predictions=self.predictions, references=self.references)
        final_score = metric.compute()
        return final_score

    def __per_prediction_score(self, wandb):
        for prediction in self.predictions:
            metric = load_metric("squad_v2")
            id_ = prediction['id']
            metric.add(prediction=prediction, reference=self.__reference_for_id(id_))
            score = metric.compute()
            wandb.log({
                'F1 per question': score[SCORE_F1],
                'Exact per question': score[SCORE_EXACT],
                'Likelihoods': self.__likelihoods_for_id(id_)
            })

    def __reference_for_id(self, ref_id: str):
        for reference in self.references:
            if reference['id'] == ref_id:
                return reference

    def __likelihoods_for_id(self, ref_id):
        result = []
        for likelihood in self.likelihoods:
            if likelihood['id'] == ref_id:
                result.append(likelihood['likelihood'])
        return result
