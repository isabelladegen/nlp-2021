# cache_dir: ./data_cache_src
# path: doc2dial
# data_ignore_verification: True
# document_dataset_name: document_domain
# document_datset_split: train
# rc_dataset_name: doc2dial_rc
# rc_training_split: train
# rc_validation_split: validation


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
