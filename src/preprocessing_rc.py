from datasets import load_dataset
from src.preprocessing_documents import preprocess_doc
from src.configurations import Configuration, QuestionPreProcessing

RC_ANSWERS = 'answers'
RC_QUESTION = 'question'
RC_DOC_ID = 'title'
RC_ID = 'id'


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


def preprocess_question(question: str, config: dict = {}) -> [str]:
    params = Configuration(**config)
    preprocess_type = params.pre_process_rc_question
    pre_processed_question = question
    if preprocess_type == QuestionPreProcessing.user_question_only.value:
        # remove history
        split_token = "agent:"
        starts_with = "user:"
        if not question.startswith(starts_with):
            print(f'question did not start with {starts_with}')
            raise
        pre_processed_question = question.split(split_token)[0].replace(starts_with, '')
    return preprocess_doc(pre_processed_question)
