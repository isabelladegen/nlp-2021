import pandas as pd
from src.preprocessing import GroundingDocument

# TODO turn into a builder test data
example_doc_id = 'Benefits Planner: Survivors | Planning'
span1_id = '6'
span1_text = 'In 2019 , '
span1 = {
    'id_sp': span1_id,
    'tag': 'u',
    'start_sp': 317,
    'end_sp': 327,
    'text_sp': span1_text,
    'title': 'Benefits Planner: Survivors | Planning For Your Survivors',
    'parent_titles': '[]',
    'id_sec': '2',
    'start_sec': 274,
    'text_sec': 'You can earn up to four credits each year. In 2019 , for example , you earn one credit for each $1,360 of wages or self - employment income. When you have earned $5,440 , you have earned your four credits for the year. ',
    'end_sec': 493
}
span2_text = 'for example , '
span2_id = '7'
span2 = {
    'id_sp': span2_id,
    'tag': 'u',
    'start_sp': 327,
    'end_sp': 341,
    'text_sp': ('%s' % span2_text),
    'title': 'Benefits Planner: Survivors | Planning For Your Survivors',
    'parent_titles': '[]',
    'id_sec': '2',
    'start_sec': 274,
    'text_sec': 'You can earn up to four credits each year. In 2019 , for example , you earn one credit for each $1,360 of wages or self - employment income. When you have earned $5,440 , you have earned your four credits for the year. ',
    'end_sec': 493
}
example_spans = [span1, span2]


def simple_data_row():
    data = {'doc_id': [example_doc_id],
            'spans': [example_spans]}
    return pd.DataFrame(data).loc[0]


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
