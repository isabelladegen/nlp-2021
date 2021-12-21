import pytest
from hamcrest import *
import pandas as pd
from src.preprocessing import *
from test_utils import *


def test_load_documents_df():
    data = load_documents_df("train")
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


def test_grounding_document_get_tagged_document_for_span_id():
    row = simple_data_row()
    document = GroundingDocument(row)
    span_ids = list(document.raw_spans.keys())

    span1_id = span_ids[0]
    span2_id = span_ids[1]
    assert_that(document.tagged_document_for(span1_id).tags, equal_to(span1_id))
    assert_that(document.tagged_document_for(span2_id).tags, equal_to(span2_id))


def test_grounding_document_gets_original_span_text_for_span_id():
    text_1 = 'some span text'
    text_2 = 'some span text'
    text_3 = 'some span text'
    span_id1 = '2'
    span_id2 = '2'
    span_id3 = '3'
    example_span1 = SpanBuilder().with_id(span_id1).with_text(text_1).build()
    example_span2 = SpanBuilder().with_id(span_id2).with_text(text_2).build()
    example_span3 = SpanBuilder().with_id(span_id3).with_text(text_3).build()
    document = GroundingDocumentBuilder().with_spans([example_span1, example_span2, example_span3]).build()

    assert_that(document.original_text_for_sp_id(span_id1), equal_to(text_1))
    assert_that(document.original_text_for_sp_id(span_id2), equal_to(text_2))
    assert_that(document.original_text_for_sp_id(span_id3), equal_to(text_3))


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
