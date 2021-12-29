from datasets import load_dataset
from src.preprocessing_documents import preprocess_doc
from src.configurations import *
import enum

RC_ANSWERS = 'answers'
RC_QUESTION = 'question'
RC_DOC_ID = 'title'
RC_ID = 'id'


# creating enumerations using class
class QuestionPreProcessing(enum.Enum):
    default = 0
    user_question_only = 1


def load_rc_dataset(split: str, config: dict = {}):
    params = Configuration(**config)
    rc_dataset = load_dataset(
        params.dataset_name,
        name=params.rc_data,
        split=split,
        ignore_verifications=params.data_ignore_verifications,
        keep_in_memory=params.keep_in_memory,
        cache_dir=params.data_cache_dir
    )
    return rc_dataset


def preprocess_question(question: str) -> [str]:
    # TODO test and do more processing
    return preprocess_doc(question)
