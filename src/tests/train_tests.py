import pytest
from src.train import TrainedModel
from src.preprocessing import preprocess_doc
from test_utils import *
from hamcrest import *


def test_trains_a_model_for_a_grounding_document():
    span_id1 = '1'
    text1 = 'span one with some text'
    span1 = SpanBuilder().with_id(span_id1).with_text(text1).build()
    span_id2 = '2'
    text2 = 'span two and some text'
    span2 = SpanBuilder().with_id(span_id2).with_text(text2).build()
    span_id3 = '3'
    text3 = 'span three and some text'
    span3 = SpanBuilder().with_id(span_id3).with_text(text3).build()

    grounding_document = GroundingDocumentBuilder().with_spans([span1, span2, span3]).build()
    trained_model = TrainedModel(grounding_document).trained_model

    assert_that(trained_model.dv[span_id1], not_none())
    assert_that(trained_model.dv[span_id2], not_none())
    assert_that(trained_model.dv[span_id3], not_none())
    assert_that(len(trained_model.dv), equal_to(3))


def test_compare_vector_with_itself():
    text = 'some span text'
    span_id = 'c'
    example_span = SpanBuilder().with_id(span_id).with_text(text).build()
    document = GroundingDocumentBuilder().with_spans([example_span]).build()

    model = TrainedModel(document).trained_model

    processed_text = preprocess_doc(text)
    vector = model.infer_vector(processed_text)
    sims = model.dv.most_similar([vector], topn=1)

    most_similar_vector = sims[0]
    assert_that(most_similar_vector[0], equal_to(span_id))
    assert_that(document.original_text_for_sp_id(span_id), equal_to(text))
    assert_that(document.processed_text_for_sp_id(span_id), equal_to(processed_text))
