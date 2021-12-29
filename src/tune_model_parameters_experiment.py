def train():
    import wandb
    from src.preprocessing_documents import load_documents_df, grounding_documents_for_dataframe
    from src.preprocessing_rc import load_rc_dataset
    from src.train_and_predict import BatchTrainer
    from src.evaluate import SCORE_F1, SCORE_EXACT, SCORE_NUMBER_OF_QUESTIONS
    from src.configurations import Configuration

    wandb.init(
        project="test",
        notes="increase vector size",
        tags=["don't train on all docs", "simple", "bigger vector", "more epochs"],
        config=Configuration().as_dict(),
        magic=True
    )

    # load document data for training
    df = load_documents_df(wandb.config)

    # get the current wand configuration in case there's a sweep running
    wand_config = Configuration(**wandb.config)

    # load limited dataset for predictions
    rc_dataset = load_rc_dataset(wand_config.sweep_rc_split, wandb.config)

    # pre-process grounding documents ready for training only for the doc ids in the rc_dataset
    grounding_documents = grounding_documents_for_dataframe(df, set(rc_dataset['doc_id']))

    # train a model for each grounding document
    trainer = BatchTrainer(grounding_documents, wandb.config)

    # predict and evaluate answers for rc dataset
    train_predictions = trainer.predict_answers_for(rc_dataset)
    train_score = train_predictions.squad2_score()

    wandb.log({
        SCORE_EXACT: train_score[SCORE_EXACT],
        SCORE_F1: train_score[SCORE_F1],
        SCORE_NUMBER_OF_QUESTIONS: train_score[SCORE_NUMBER_OF_QUESTIONS],
    })
