from typing import Any, List
from datasets import Dataset
import random
from gensim.models import KeyedVectors
from gensim.models.doc2vec import Doc2Vec

from src.preprocessing_documents import GroundingDocument
from src.preprocessing_rc import *
from src.evaluate import PredictionsEvaluation
from src.configurations import Configuration


class TrainedModel:
    hyperparams: Configuration
    id: str  # doc id
    grounding_doc: GroundingDocument
    trained_model: Doc2Vec

    def __init__(self, grounding_doc: GroundingDocument, hyperparams: Configuration = None):
        if not hyperparams:
            hyperparams = Configuration()
        self.hyperparams = hyperparams
        self.grounding_doc = grounding_doc
        self.id = grounding_doc.id
        self.trained_model = self.__train_model()

    def __train_model(self):
        documents = self.grounding_doc.tagged_documents
        return Doc2Vec(documents,
                       vector_size=self.hyperparams.vector_size,
                       window=self.hyperparams.window,
                       min_count=self.hyperparams.min_count,
                       workers=self.hyperparams.workers,
                       dm=self.hyperparams.dm,
                       epochs=self.hyperparams.epochs)

    def document_vectors(self) -> KeyedVectors:
        return self.trained_model.dv

    def get_n_most_similar_vectors(self, processed_doc: [str], n: int) -> dict[str, float]:
        vector_for_new_doc = self.trained_model.infer_vector(processed_doc)
        sims = self.trained_model.dv.most_similar([vector_for_new_doc], topn=n)
        return dict(sims)

    def predict_grounding_text_for(self, preprocessed_question: [str]):
        most_likely = self.get_n_most_similar_vectors(preprocessed_question,
                                                      self.hyperparams.number_of_most_likely_docs)
        grounding_text = ''
        likelihoods: list[float] = []
        for span_id in sorted(list(most_likely.keys())):
            grounding_text = grounding_text + self.grounding_doc.original_text_for_sp_id(span_id) + ' '
            likelihoods.append(most_likely[span_id])

        return grounding_text.strip(), likelihoods


class BatchTrainer:
    trainedModels: dict[str, TrainedModel]
    grounding_docs: [GroundingDocument]

    def __init__(self, grounding_docs: [GroundingDocument], hyperparams: dict = {}):
        config = Configuration(**hyperparams)
        self.hyperparams = config
        self.grounding_docs = grounding_docs
        self.trainedModels = self.__trained_models()

    def model_for_doc_id(self, grounding_doc_id: str) -> TrainedModel:
        return self.trainedModels[grounding_doc_id]

    def __trained_models(self) -> dict[Any, TrainedModel]:
        result: dict[Any, TrainedModel] = {}
        for doc in self.grounding_docs:
            result[doc.id] = TrainedModel(doc, self.hyperparams)
        return result

    def predict_answers_for(self, rc_dataset: Dataset) -> PredictionsEvaluation:
        predictions_evaluation = PredictionsEvaluation()
        for row in rc_dataset:
            prediction_id = row[RC_ID]
            question_ = row[RC_QUESTION]
            doc_id = row[RC_DOC_ID]
            gold_answer = row[RC_ANSWERS]

            # process question
            preprocessed_question = preprocess_question(question_, self.hyperparams)
            # get trained model
            model = self.model_for_doc_id(doc_id)
            # get prediction
            most_likely_answer, likelihood = model.predict_grounding_text_for(preprocessed_question)

            # for evaluation combine predicted answer and gold answer
            predictions_evaluation.add(prediction_id, most_likely_answer, gold_answer, likelihood)

        return predictions_evaluation

    def predict_random_spans_for(self, rc_dataset: Dataset) -> PredictionsEvaluation:
        predictions_evaluation = PredictionsEvaluation()
        for row in rc_dataset:
            prediction_id = row[RC_ID]
            doc_id = row[RC_DOC_ID]
            gold_answer = row[RC_ANSWERS]

            # pick random span
            model = self.model_for_doc_id(doc_id)
            spans = list(model.grounding_doc.raw_docs.values())
            random_answer = random.choice(spans)
            predictions_evaluation.add(prediction_id, random_answer, gold_answer)

        return predictions_evaluation
