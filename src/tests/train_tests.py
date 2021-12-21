import pytest
from src.preprocessing import GroundingDocument
from src.train import TrainedModel
import pandas as pd
from src.preprocessing import preprocess_doc
from test_utils import *
from hamcrest import *


def test_trains_a_model_for_a_grounding_document():
    grounding_document = GroundingDocument(simple_data_row())
    model = TrainedModel(grounding_document)

    assert len(model.trained_model.dv) == len(grounding_document.raw_spans)


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

