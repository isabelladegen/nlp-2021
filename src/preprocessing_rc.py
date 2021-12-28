from datasets import load_dataset
from src.preprocessing_documents import preprocess_doc

RC_ANSWERS = 'answers'
RC_QUESTION = 'question'
RC_DOC_ID = 'title'
RC_ID = 'id'


def load_rc_dataset(split: str):
    rc_dataset = load_dataset(
        "doc2dial",
        name="doc2dial_rc",
        split=split,
        ignore_verifications=True,
        cache_dir="./data_cache_src"
    )
    return rc_dataset


def preprocess_question(question: str) -> [str]:
    # TODO test and do more processing
    return preprocess_doc(question)
