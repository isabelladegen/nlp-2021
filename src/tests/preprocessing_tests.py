import pytest
import pandas as pd
from src.preprocessing import load_documents_df
from src.preprocessing import GroundingDocument
from src.preprocessing import preprocess_doc
from src.preprocessing import grounding_documents_for_dataframe

# test data
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


def test_load_documents_df():
    data = load_documents_df("train")

    # print(data.loc[0]['spans'])
    # print(type(data.log[0]))
    assert data.shape == (488, 3), ''


def test_preprocess_doc_returns_tokenzied_string():
    example3_raw_text = "You can earn up to four credits each year. In 2019 , for example , you earn one credit for each $1,360 of wages or self - employment income."
    example1 = preprocess_doc(span1_text)
    example2 = preprocess_doc(span2_text)
    example3 = preprocess_doc(example3_raw_text)
    assert example1 == ['in']
    assert example2 == ['for', 'example']
    assert example3 == ['you', 'can', 'earn', 'up', 'to', 'four', 'credits', 'each', 'year', 'in', 'for', 'example',
                        'you', 'earn', 'one', 'credit', 'for', 'each', 'of', 'wages', 'or', 'self', 'employment',
                        'income']


def test_grounding_document_reads_document_id():
    row = simple_data_row()
    document = GroundingDocument(row)

    assert row['doc_id'] == example_doc_id
    assert document.id == example_doc_id


def test_grounding_document_reads_spans():
    row = simple_data_row()
    document = GroundingDocument(row)

    raw_spans = document.raw_spans
    assert len(raw_spans) == len(example_spans)
    assert raw_spans[span1_id] == span1_text
    assert raw_spans[span2_id] == span2_text


def test_grounding_document_preprocesses_span_text():
    row = simple_data_row()
    document = GroundingDocument(row)

    preprocessed_spans = document.preprocessed_spans
    assert len(preprocessed_spans) == len(example_spans)
    assert preprocessed_spans[span1_id] == preprocess_doc(span1_text)
    assert preprocessed_spans[span2_id] == preprocess_doc(span2_text)


def test_grounding_document_creates_tagged_documents():
    row = simple_data_row()
    document = GroundingDocument(row)

    tagged_documents = document.tagged_documents
    assert len(tagged_documents) == len(example_spans)
    tagged_doc1 = tagged_documents[0]
    tagged_doc2 = tagged_documents[1]
    assert tagged_doc1.tags == span1_id
    assert tagged_doc2.tags == span2_id
    assert tagged_doc1.words == preprocess_doc(span1_text)
    assert tagged_doc2.words == preprocess_doc(span2_text)


def test_get_grounding_documents_for_dataset():
    df = load_documents_df("train")
    example = df.loc[100]

    grounding_documents = grounding_documents_for_dataframe(df)
    assert len(grounding_documents) == df.shape[0]
    example_grounding_doc = grounding_documents[100]
    assert example_grounding_doc.id == example['doc_id']
    assert len(example_grounding_doc.raw_spans) == len(example['spans'])


# Utils methods - TODO move out
def simple_data_row():
    data = {'doc_id': [example_doc_id],
            'spans': [example_spans]}
    return pd.DataFrame(data).loc[0]
