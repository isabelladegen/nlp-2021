import pandas as pd
from src.preprocessing import GroundingDocument


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
        self.spans = SpanBuilder().build()

    def build(self):
        data = {'doc_id': [id],
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
