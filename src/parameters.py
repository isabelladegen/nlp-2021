# cache_dir: ./data_cache_src
# path: doc2dial
# data_ignore_verification: True
# document_dataset_name: document_domain
# document_datset_split: train
# rc_dataset_name: doc2dial_rc
# rc_training_split: train
# rc_validation_split: validation


config_params = dict(
    training_split="train",
    validation_split="validation",
    dataset_name="doc2dial",
    document_data_name="document_domain",
    data_ignore_verifications=True,
    data_cache_dir="./data_cache_src"
)
