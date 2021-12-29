from dataclasses import dataclass, asdict


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
    sweep_rc_split: str = 'train[:10%]'

    # Doc2Vec model parameters
    vector_size: int = 10
    window: int = 4
    min_count: int = 1
    workers: int = 4
    dm: int = 1
    epochs: int = 30

    def as_dict(self):
        return asdict(self)


config_params = dict(
    # Dataset Params
    dataset_name="doc2dial",
    document_data_name="document_domain",
    rc_data="doc2dial_rc",
    data_ignore_verifications=True,
    data_cache_dir="./data_cache_src",
    keep_in_memory=True,
    # Model Params
    vector_size=10,
    window=4,
    min_count=1,
    workers=4,
    dm=1,
    epochs=30
)
