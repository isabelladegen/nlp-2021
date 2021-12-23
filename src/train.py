from typing import Dict, Any

from gensim.models import KeyedVectors

from src.preprocessing import GroundingDocument
from gensim.models.doc2vec import Doc2Vec


def train_models_for(grounding_docs: [GroundingDocument]):
    return []


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
