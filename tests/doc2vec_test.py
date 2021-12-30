from hamcrest import *
from gensim.models.doc2vec import Doc2Vec, TaggedDocument


# hugely important, if tags are not given as a list, the string of each tag will be split into multiple tags and
# you'll end up with more vectors than you thought
def test_tagged_document_behaviour():
    tag1 = 'tag1'
    tagged_document1 = TaggedDocument(['some', 'tokened', 'doc', 'just', 'to', 'test'], [tag1])
    tag2 = 'tag2'
    tagged_document2 = TaggedDocument(['some', 'more', 'test', 'to', 'make', 'it', 'interesting'], [tag2])
    documents = [tagged_document1, tagged_document2]

    model = Doc2Vec(documents, vector_size=4, window=4, min_count=1, workers=1, epochs=1)

    assert_that(len(model.dv), equal_to(2))
    assert_that(model.dv[tag1], not_none())
    assert_that(model.dv[tag2], not_none())
