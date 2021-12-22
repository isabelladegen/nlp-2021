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
                       epochs=30)
