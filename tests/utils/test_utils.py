import pandas as pd
from datasets import Dataset
from src.preprocessing_documents import GroundingDocument
from src.preprocessing_rc import *


class DocumentDatasetBuilder:
    def __init__(self):
        span = SpanBuilder().build()
        doc_id = span['title']
        self.doc_id = doc_id
        self.spans = [span]

    def build(self):
        # this imitates the dataset from huggingface
        data = {'doc_id': [self.doc_id],
                'spans': [self.spans]}
        return pd.DataFrame(data).loc[0]

    def with_spans(self, spans: [{}]):
        """

        :type spans: list of span dictionaries
        """
        self.spans = spans
        return self

    def with_doc_id(self, doc_id: str):
        self.doc_id = doc_id
        return self


class GroundingDocumentBuilder:
    def __init__(self):
        self.id = "Some document id"
        self.spans = [SpanBuilder().build()]

    def build(self):
        data = {'doc_id': [self.id],
                'spans': [self.spans]}
        return GroundingDocument(pd.DataFrame(data).loc[0])  # TODO get rid of dataframe and just use dataset

    def with_spans(self, spans: [{}]):
        self.spans = spans
        return self

    def with_doc_id(self, doc_id: str):
        self.id = doc_id
        return self


class SpanBuilder:
    def __init__(self):
        self.text = "Some generic span text serving as doc"
        self.id = 'some id'

    def build(self):
        return {
            'id_sp': self.id,
            'tag': 'u',
            'start_sp': 327,
            'end_sp': 341,
            'text_sp': self.text,
            'title': 'some document',
            'parent_titles': '[]',
            'id_sec': '2',
            'start_sec': 274,
            'text_sec': 'Some test',
            'end_sec': 493
        }

    def with_id(self, span_id: str):
        self.id = span_id
        return self

    def with_text(self, span_text: str):
        self.text = span_text
        return self


class RCDatasetBuilder:
    def __init__(self):
        self.ids = []
        self.doc_ids = []
        self.questions = []
        self.answers = []

    def build(self) -> Dataset:
        if not self.ids:
            self.__create_default_data()
        data = {RC_ID: self.ids,
                RC_DOC_ID: self.doc_ids,
                RC_QUESTION: self.questions,
                RC_ANSWERS: self.answers
                }
        return Dataset.from_dict(data)

    def with_question_answer(self, question_id, document_id, question_text, answer_text):
        self.ids.append(question_id)
        self.doc_ids.append(document_id)
        self.questions.append(question_text)
        self.answers.append(ReferenceAnswerBuilder().with_id(question_id).with_text(answer_text).build())
        return self

    def __create_default_data(self):
        id1 = '1'
        id2 = '2'
        self.ids = [id1, id2]
        self.doc_ids = ['doc1', 'doc2']
        self.questions = ['question1', 'question2']
        answer1 = ReferenceAnswerBuilder().with_id(id1).with_text("this is the answer to question 1").build()
        answer2 = ReferenceAnswerBuilder().with_id(id2).with_text("this is the answer to question 2").build()
        self.answers = [answer1, answer2]


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
            'text': [
                self.answer_text
            ],
            'answer_start': [
                self.answer_start
            ]
        }
        return squad2_metric_answer

    def with_id(self, answer_id):
        self.id = answer_id
        return self

    def with_text(self, answer_text):
        self.answer_text = answer_text
        return self
