from typing import Dict, Any

import pandas as pd
from gensim.models import KeyedVectors

from src.preprocessing_documents import GroundingDocument
from gensim.models.doc2vec import Doc2Vec
from src.preprocessing_rc import *
from datasets import Dataset


class TrainedModel:
    id: str  # doc id
    grounding_doc: GroundingDocument
    trained_model: Doc2Vec

    def __init__(self, grounding_doc: GroundingDocument):
        self.grounding_doc = grounding_doc
        self.id = grounding_doc.id
        self.trained_model = self.__train_model()

    def __train_model(self):
        # TODO log model config
        documents = self.grounding_doc.tagged_documents
        return Doc2Vec(documents,
                       vector_size=10,
                       window=4,
                       min_count=1,
                       workers=4,
                       dm=1,
                       epochs=30)

    def document_vectors(self) -> KeyedVectors:
        return self.trained_model.dv

    def get_n_most_similar_vectors(self, processed_doc: [str], n: int) -> dict[str, float]:
        vector_for_new_doc = self.trained_model.infer_vector(processed_doc)
        sims = self.trained_model.dv.most_similar([vector_for_new_doc], topn=n)
        return dict(sims)

    def predict_grounding_text_for(self, preprocessed_question: [str]) -> str:
        most_likely = self.get_n_most_similar_vectors(preprocessed_question, 1)
        # TODO make number of texts that are combined configurable
        return self.grounding_doc.original_text_for_sp_id(list(most_likely.keys())[0])


class BatchTrainer:
    trainedModels: dict[str, TrainedModel]
    grounding_docs: [GroundingDocument]

    def __init__(self, grounding_docs: [GroundingDocument]):
        self.grounding_docs = grounding_docs
        self.trainedModels = self.__trained_models()

    def model_for_doc_id(self, grounding_doc_id: str) -> TrainedModel:
        return self.trainedModels[grounding_doc_id]

    def __trained_models(self) -> dict[Any, TrainedModel]:
        result: dict[Any, TrainedModel] = {}
        for doc in self.grounding_docs:
            result[doc.id] = TrainedModel(doc)
        return result

    def predictions_and_referenced_for(self, rc_dataset: Dataset):
        predictions = []
        references = []
        for row in rc_dataset:
            # process question
            question = preprocess_question(row[RC_QUESTION])
            doc_id = row[RC_DOC_ID]

            # get trained model
            model = self.model_for_doc_id(doc_id)
            most_likely_answer = model.predict_grounding_text_for(question)

            id_ = row["id"]
            predictions.append(
                {'id': id_,
                 'prediction_text':
                     most_likely_answer,
                 'no_answer_probability': 0.0
                 }
            )

            # just using their answers
            references.append(
                {
                    "id": id_,
                    "answers": row["answers"],
                }
            )
        return predictions, references
