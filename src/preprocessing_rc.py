from datasets import load_dataset
from src.preprocessing_documents import preprocess_doc
from src.parameters import *

RC_ANSWERS = 'answers'
RC_QUESTION = 'question'
RC_DOC_ID = 'title'
RC_ID = 'id'


def load_rc_dataset(split: str, config: dict = None):
    if not config:
        config = config_params
    rc_dataset = load_dataset(
        config['dataset_name'],
        name=config['rc_data'],
        split=split,
        ignore_verifications=config['data_ignore_verifications'],
        keep_in_memory=config['keep_in_memory'],
        cache_dir=config['data_cache_dir']
    )
    return rc_dataset


def preprocess_question(question: str) -> [str]:
    # TODO test and do more processing
    return preprocess_doc(question)
