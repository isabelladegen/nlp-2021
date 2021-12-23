import pytest
from hamcrest import *
from src.preprocessing_documents import *
from test_utils import *


def test_load_documents_df():
    data = load_documents_df("train")
    assert data.shape == (488, 3), ''


def test_preprocess_doc_returns_tokenized_string():
    example3_raw_text = "You can earn up to four credits each year. In 2019 , for example , you earn one " \
                        "credit for each $1,360 of wages or self - employment income."
    lower_case_no_apostorphies_nodashwords = preprocess_doc("SOME other text1 with doesn't examples and bind-words")
    numbers_removed = preprocess_doc("1, 2, 3")
    example3 = preprocess_doc(example3_raw_text)
    assert lower_case_no_apostorphies_nodashwords == ['some', 'other', 'text', 'with', 'doesn', 'examples', 'and',
                                                      'bind', 'words']
    assert numbers_removed == []
    assert example3 == ['you', 'can', 'earn', 'up', 'to', 'four', 'credits', 'each', 'year', 'in', 'for', 'example',
                        'you', 'earn', 'one', 'credit', 'for', 'each', 'of', 'wages', 'or', 'self', 'employment',
                        'income']


def test_grounding_document_reads_document_id():
    doc_id = "some test document id"
    span = SpanBuilder().build()
    pd_row = DocumentDatasetBuilder() \
        .with_doc_id(doc_id) \
        .with_spans([span]) \
        .build()

    document = GroundingDocument(pd_row)

    assert_that(document.id, equal_to(doc_id))


def test_grounding_document_reads_spans_for_each_doc():
    # build a document with three spans
    doc_id = "first doc"
    span_id1 = 'first span'
    text1 = 'span one with some text'
    span1 = SpanBuilder().with_id(span_id1).with_text(text1).build()
    span_id2 = 'second span'
    text2 = 'span two and some text'
    span2 = SpanBuilder().with_id(span_id2).with_text(text2).build()
    span_id3 = 'third span'
    text3 = 'span three and some text'
    span3 = SpanBuilder().with_id(span_id3).with_text(text3).build()
    pd_row = DocumentDatasetBuilder() \
        .with_doc_id(doc_id) \
        .with_spans([span1, span2, span3]) \
        .build()

    document = GroundingDocument(pd_row)

    raw_spans = document.raw_spans
    assert_that(document.id, equal_to(doc_id))
    assert_that(raw_spans, has_length(3))
    assert_that(raw_spans[span_id1], equal_to(text1))
    assert_that(raw_spans[span_id2], equal_to(text2))
    assert_that(raw_spans[span_id3], equal_to(text3))


def test_grounding_document_get_tagged_document_for_span_id():
    # build a document with three spans
    doc_id = "first doc"
    span_id1 = 'first span'
    text1 = 'span one with some text'
    span1 = SpanBuilder().with_id(span_id1).with_text(text1).build()
    span_id2 = 'second span'
    text2 = 'span two and some text'
    span2 = SpanBuilder().with_id(span_id2).with_text(text2).build()
    span_id3 = 'third span'
    text3 = 'span three and some text'
    span3 = SpanBuilder().with_id(span_id3).with_text(text3).build()
    pd_row = DocumentDatasetBuilder() \
        .with_doc_id(doc_id) \
        .with_spans([span1, span2, span3]) \
        .build()

    document = GroundingDocument(pd_row)
    tagged_doc1 = document.tagged_document_for(span_id1)
    assert_that(tagged_doc1.tags[0], equal_to(span_id1))
    assert_that(tagged_doc1.words, equal_to(preprocess_doc(text1)))

    tagged_doc2 = document.tagged_document_for(span_id2)
    assert_that(tagged_doc2.tags[0], equal_to(span_id2))
    assert_that(tagged_doc2.words, equal_to(preprocess_doc(text2)))

    tagged_doc3 = document.tagged_document_for(span_id3)
    assert_that(tagged_doc3.tags[0], equal_to(span_id3))
    assert_that(tagged_doc3.words, equal_to(preprocess_doc(text3)))


def test_grounding_document_gets_original_span_text_for_span_id():
    # build a document with three spans
    doc_id = "first doc"
    span_id1 = 'first span'
    text1 = 'span one with some text'
    span1 = SpanBuilder().with_id(span_id1).with_text(text1).build()
    span_id2 = 'second span'
    text2 = 'span two and some text'
    span2 = SpanBuilder().with_id(span_id2).with_text(text2).build()
    span_id3 = 'third span'
    text3 = 'span three and some text'
    span3 = SpanBuilder().with_id(span_id3).with_text(text3).build()
    pd_row = DocumentDatasetBuilder() \
        .with_doc_id(doc_id) \
        .with_spans([span1, span2, span3]) \
        .build()

    document = GroundingDocument(pd_row)

    assert_that(document.original_text_for_sp_id(span_id1), equal_to(text1))
    assert_that(document.original_text_for_sp_id(span_id2), equal_to(text2))
    assert_that(document.original_text_for_sp_id(span_id3), equal_to(text3))


def test_grounding_document_preprocesses_span_text():
    # build a document with three spans
    doc_id = "first doc"
    span_id1 = 'first span'
    text1 = 'span one with some text'
    span1 = SpanBuilder().with_id(span_id1).with_text(text1).build()
    span_id2 = 'second span'
    text2 = 'span two and some text'
    span2 = SpanBuilder().with_id(span_id2).with_text(text2).build()
    span_id3 = 'third span'
    text3 = 'span three and some text'
    span3 = SpanBuilder().with_id(span_id3).with_text(text3).build()
    pd_row = DocumentDatasetBuilder() \
        .with_doc_id(doc_id) \
        .with_spans([span1, span2, span3]) \
        .build()

    document = GroundingDocument(pd_row)

    preprocessed_spans = document.preprocessed_spans
    assert_that(preprocessed_spans, has_length(3))
    assert_that(preprocessed_spans[span_id1], equal_to(preprocess_doc(text1)))
    assert_that(preprocessed_spans[span_id2], equal_to(preprocess_doc(text2)))
    assert_that(preprocessed_spans[span_id3], equal_to(preprocess_doc(text3)))


def test_grounding_document_creates_tagged_documents():
    # build a document with three spans
    doc_id = "first doc"
    span_id1 = 'first span'
    text1 = 'span one with some text'
    span1 = SpanBuilder().with_id(span_id1).with_text(text1).build()
    span_id2 = 'second span'
    text2 = 'span two and some text'
    span2 = SpanBuilder().with_id(span_id2).with_text(text2).build()
    span_id3 = 'third span'
    text3 = 'span three and some text'
    span3 = SpanBuilder().with_id(span_id3).with_text(text3).build()
    pd_row = DocumentDatasetBuilder() \
        .with_doc_id(doc_id) \
        .with_spans([span1, span2, span3]) \
        .build()

    document = GroundingDocument(pd_row)

    tagged_documents = document.tagged_documents
    assert_that(tagged_documents, has_length(3))
    assert_that(tagged_documents[0].tags[0], equal_to(span_id1))
    assert_that(tagged_documents[1].tags[0], equal_to(span_id2))
    assert_that(tagged_documents[2].tags[0], equal_to(span_id3))
    assert_that(tagged_documents[0].words, equal_to(preprocess_doc(text1)))
    assert_that(tagged_documents[1].words, equal_to(preprocess_doc(text2)))
    assert_that(tagged_documents[2].words, equal_to(preprocess_doc(text3)))


def test_get_grounding_documents_for_huggingface_document_dataset():
    df = load_documents_df("train")
    example_row = df.loc[100]

    grounding_documents = grounding_documents_for_dataframe(df)
    assert_that(grounding_documents, has_length(df.shape[0]))

    example_grounding_doc = grounding_documents[100]
    assert_that(example_grounding_doc.id, equal_to(example_row['doc_id']))
    assert_that(len(example_grounding_doc.raw_spans), equal_to(len(example_row['spans'])))
