from hamcrest import *
from src.train_and_predict import TrainedModel, BatchTrainer
from src.preprocessing_documents import load_documents_df, grounding_documents_for_dataframe
from src.configurations import Configuration
from utils.test_utils import *


def test_trains_a_model_for_a_grounding_document():
    span_id1 = "first span id"
    text1 = 'span one with some text'
    span1 = SpanBuilder().with_id(span_id1).with_text(text1).build()
    span_id2 = "second span id"
    text2 = 'span two and some text'
    span2 = SpanBuilder().with_id(span_id2).with_text(text2).build()
    span_id3 = "3"
    text3 = 'span three and some text'
    span3 = SpanBuilder().with_id(span_id3).with_text(text3).build()

    grounding_document = GroundingDocumentBuilder().with_spans([span1, span2, span3]).build()
    trained_model = TrainedModel(grounding_document).trained_model

    assert_that(len(trained_model.dv), equal_to(3))
    assert_that(trained_model.dv[span_id1], not_none())
    assert_that(trained_model.dv[span_id2], not_none())
    assert_that(trained_model.dv[span_id3], not_none())


def test_predicts_the_same_vector_to_be_most_similar():
    text = 'some span text'
    span_id = 'c'
    example_span = SpanBuilder().with_id(span_id).with_text(text).build()
    document = GroundingDocumentBuilder().with_spans([example_span]).build()

    model = TrainedModel(document)
    processed_text = preprocess_doc(text)
    most_similar = model.get_n_most_similar_vectors(processed_text, 1)
    assert_that(most_similar[span_id], not_none())
    assert_that(document.original_text_for_sp_id(span_id), equal_to(text))
    assert_that(document.processed_text_for_sp_id(span_id), equal_to(processed_text))


def test_predicts_grounding_text_for_given_question():
    # setup grounding doc
    text = 'some span text'
    span_id = 'c'
    example_span = SpanBuilder().with_id(span_id).with_text(text).build()
    document = GroundingDocumentBuilder().with_spans([example_span]).build()

    # setup user question
    question = "what is a span?"
    processed_question = preprocess_doc(question)

    # train model
    model = TrainedModel(document)

    # predict most likely grounding text
    most_likely_grounding_text, likelihood = model.predict_grounding_text_for(processed_question)

    assert_that(most_likely_grounding_text, equal_to(text))


def test_predicts_n_grounding_texts_for_a_given_question_sorted_by_span_id():
    # setup grounding doc
    span_text1 = 'Some span text.'
    span_id1 = 'span2'
    span1 = SpanBuilder().with_id(span_id1).with_text(span_text1).build()
    span_text2 = 'Some new span that could be returned too as likely.'
    span_id2 = 'span1'
    span2 = SpanBuilder().with_id(span_id2).with_text(span_text2).build()
    document = GroundingDocumentBuilder().with_spans([span1, span2]).build()

    # setup user question
    question = "what is a span?"
    processed_question = preprocess_doc(question)

    # train model
    config = Configuration()
    config.number_of_most_likely_docs = 2
    model = TrainedModel(document, config)

    # predict most likely grounding text
    most_likely_grounding_text, likelihoods = model.predict_grounding_text_for(processed_question)

    assert_that(most_likely_grounding_text, equal_to(span_text2 + ' ' + span_text1))
    assert_that(len(likelihoods), equal_to(config.number_of_most_likely_docs))


def test_return_own_vector_for_most_similar_vectors():
    span_id1 = "first span id"
    text1 = 'one lets get started Paris geography'
    span1 = SpanBuilder().with_id(span_id1).with_text(text1).build()
    span_id2 = "second span id"
    text2 = 'two powerful'
    span2 = SpanBuilder().with_id(span_id2).with_text(text2).build()
    span_id3 = "3"
    text3 = 'three something totally different'
    span3 = SpanBuilder().with_id(span_id3).with_text(text3).build()

    grounding_document = GroundingDocumentBuilder().with_spans([span1, span2, span3]).build()
    trained_model = TrainedModel(grounding_document)

    n = 1
    processed_text1 = preprocess_doc(text1)
    most_similar_to_span_1 = trained_model.get_n_most_similar_vectors(processed_text1, n)
    three_most_similar_to_span_1 = trained_model.get_n_most_similar_vectors(processed_text1,
                                                                            len(grounding_document.raw_docs))
    most_similar_to_span_2 = trained_model.get_n_most_similar_vectors(preprocess_doc(text2), n)
    most_similar_to_span_3 = trained_model.get_n_most_similar_vectors(preprocess_doc(text3), n)

    assert_that(len(most_similar_to_span_1), equal_to(n))
    assert_that(len(three_most_similar_to_span_1), equal_to(len(grounding_document.raw_docs)))
    # returns vector for span 1
    print(f'First: {most_similar_to_span_1}')
    print(f'Second: {most_similar_to_span_2}')
    print(f'Third: {most_similar_to_span_3}')
    # TODO think about to be able to do this with small documents if at all
    # assert_that(most_similar_to_span_1[span_id1], greater_than(0.0))
    # assert_that(most_similar_to_span_2[span_id2], greater_than(0.0))
    # assert_that(most_similar_to_span_3[span_id2], greater_than(0.0))


def test_trains_a_model_for_each_document():
    df = load_documents_df()
    smaller_dataset = df.head(10)

    grounding_documents = grounding_documents_for_dataframe(smaller_dataset)
    assert_that(grounding_documents, has_length(smaller_dataset.shape[0]))

    # test models for two different grounding documents
    example1 = grounding_documents[5]
    example2 = grounding_documents[2]
    trainer = BatchTrainer(grounding_documents)
    trained_model1 = trainer.model_for_doc_id(example1.id)
    trained_model2 = trainer.model_for_doc_id(example2.id)

    # number of vectors = number of spans in grounding document
    assert_that(len(trained_model1.document_vectors()), equal_to(len(example1.raw_docs)))
    assert_that(len(trained_model2.document_vectors()), equal_to(len(example2.raw_docs)))


def test_returns_predictions_and_references():
    span_id1 = "first span id"
    text1 = 'one lets get started Paris geography'
    span1 = SpanBuilder().with_id(span_id1).with_text(text1).build()
    span_id2 = "second span id"
    text2 = 'two powerful'
    span2 = SpanBuilder().with_id(span_id2).with_text(text2).build()
    span_id3 = "3"
    text3 = 'three something totally different'
    span3 = SpanBuilder().with_id(span_id3).with_text(text3).build()
    grounding_document = GroundingDocumentBuilder().with_spans([span1, span2, span3]).build()

    rc_dataset = RCDatasetBuilder() \
        .with_question_answer("rcId1", grounding_document.id, "user:Paris question", text1) \
        .with_question_answer("rcId2", grounding_document.id, "user:Power question", text2) \
        .with_question_answer("rcId3", grounding_document.id, "user:Different question", text3) \
        .with_question_answer("rcId4", grounding_document.id, "user:Another question", text3) \
        .build()

    # train
    trainer = BatchTrainer([grounding_document])

    # get predictions and gold answers
    prediction_evaluation = trainer.predict_answers_for(rc_dataset)

    ids = rc_dataset[RC_ID]
    predictions = prediction_evaluation.predictions
    references = prediction_evaluation.references
    assert_that(len(predictions), equal_to(rc_dataset.num_rows))
    assert_that(len(references), equal_to(rc_dataset.num_rows))
    assert_that(predictions[0]['id'], equal_to(ids[0]))
    assert_that(references[1]['id'], equal_to(ids[1]))
    assert_that(references[2]['id'], equal_to(ids[2]))
    assert_that(references[3]['id'], equal_to(ids[3]))


def test_predicts_random_span_for_a_document():
    span_id1 = "first span id"
    text1 = 'one lets get started Paris geography'
    span1 = SpanBuilder().with_id(span_id1).with_text(text1).build()
    span_id2 = "second span id"
    text2 = 'two powerful'
    span2 = SpanBuilder().with_id(span_id2).with_text(text2).build()
    span_id3 = "3"
    text3 = 'three something totally different'
    span3 = SpanBuilder().with_id(span_id3).with_text(text3).build()
    grounding_document = GroundingDocumentBuilder().with_spans([span1, span2, span3]).build()

    rc_dataset = RCDatasetBuilder() \
        .with_question_answer("rcId1", grounding_document.id, "Paris question", text1) \
        .with_question_answer("rcId2", grounding_document.id, "Power question", text2) \
        .with_question_answer("rcId3", grounding_document.id, "Different question", text3) \
        .with_question_answer("rcId4", grounding_document.id, "Another question", text3) \
        .build()

    trainer = BatchTrainer([grounding_document])
    random_predictions = trainer.predict_random_spans_for(rc_dataset)

    assert_that(len(random_predictions.predictions), equal_to(rc_dataset.num_rows))
    assert_that(len(random_predictions.references), equal_to(rc_dataset.num_rows))
    # noinspection PyTypeChecker
    assert_that(list(grounding_document.raw_docs.values()),
                has_item(random_predictions.predictions[0]['prediction_text']))
    # noinspection PyTypeChecker
    assert_that(list(grounding_document.raw_docs.values()),
                has_item(random_predictions.predictions[1]['prediction_text']))
    # noinspection PyTypeChecker
    assert_that(list(grounding_document.raw_docs.values()),
                has_item(random_predictions.predictions[2]['prediction_text']))
    # noinspection PyTypeChecker
    assert_that(list(grounding_document.raw_docs.values()),
                has_item(random_predictions.predictions[3]['prediction_text']))
