from dataclasses import dataclass, asdict
import enum


# creating enumerations using class
class QuestionPreProcessing(enum.Enum):
    default = 0
    user_question_only = 1  # provide no context just the user question


@dataclass
class Configuration:
    # hugging face dataset general
    dataset_name: str = 'doc2dial'
    data_ignore_verifications: bool = True
    data_cache_dir: str = './data_cache_src'
    keep_in_memory: bool = True

    # document dataset
    document_data_split: str = 'train'
    document_data_name: str = 'document_domain'

    # training and validation question/answer dataset
    rc_data: str = 'doc2dial_rc'

    # what rc split to use for various training
    predict_answers_rc_split: str = 'train'
    validate_answers_rc_split: str = 'validation'
    random_answers_rc_split: str = 'train'
    sweep_rc_split: str = 'train[:20%]'

    # preprocess user questions
    pre_process_rc_question: int = QuestionPreProcessing.user_question_only.value

    # Doc2Vec model parameters
    vector_size: int = 100
    window: int = 4
    min_count: int = 1
    workers: int = 4
    dm: int = 0
    epochs: int = 150

    # predictions parameters
    number_of_most_likely_docs: int = 3

    def as_dict(self):
        return asdict(self)
