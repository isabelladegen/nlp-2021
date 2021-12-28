def train():
    import wandb
    from src.preprocessing_documents import load_documents_df, grounding_documents_for_dataframe
    from src.preprocessing_rc import load_rc_dataset
    from src.train_and_predict import BatchTrainer
    from src.evaluate import PredictionsEvaluation, SCORE_F1, SCORE_EXACT, SCORE_NUMBER_OF_QUESTIONS

    defaults = dict(
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

    wandb.init(
        project="test",
        notes="increase vector size",
        tags=["simple", "bigger vector", "more epochs"],
        config=defaults,
        magic=True
    )

    # load document data for training
    df = load_documents_df(wandb.config)

    # pre-process grounding documents ready for training
    grounding_documents = grounding_documents_for_dataframe(df)

    # train a model for each grounding document
    trainer = BatchTrainer(grounding_documents, wandb.config)

    rc_dataset = load_rc_dataset("validation", wandb.config)
    train_predictions = trainer.predict_answers_for(rc_dataset)

    train_score = train_predictions.squad2_score()

    wandb.log({
        SCORE_EXACT: train_score[SCORE_EXACT],
        SCORE_F1: train_score[SCORE_F1],
        SCORE_NUMBER_OF_QUESTIONS: train_score[SCORE_NUMBER_OF_QUESTIONS],
    })
