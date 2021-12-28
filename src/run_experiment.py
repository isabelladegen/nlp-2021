import wandb

import src.parameters
from src.preprocessing_documents import load_documents_df, grounding_documents_for_dataframe
from src.preprocessing_rc import load_rc_dataset
from src.train_and_predict import BatchTrainer
from src.evaluate import *

wandb.init(
    project="test",
    notes="figure out what to log and configure",
    tags=["simple", "more logs and configs"],
    config=src.parameters.config_params
)

# load document data for training
df = load_documents_df(wandb.config)

# pre-process grounding documents ready for training
grounding_documents = grounding_documents_for_dataframe(df)

# train a model for each grounding document
trainer = BatchTrainer(grounding_documents, config_params)

train_predictions = trainer.predict_answers_for(load_rc_dataset("train", wandb.config))

train_score = train_predictions.squad2_score()

validation_predictions = trainer.predict_answers_for(load_rc_dataset("validation", wandb.config))
validation_score = validation_predictions.squad2_score()

random_validation_predictions = trainer.predict_random_spans_for(load_rc_dataset("validation", wandb.config))
random_validation_score = validation_predictions.squad2_score()

wandb.log({
    SCORE_EXACT: train_score[SCORE_EXACT],
    SCORE_F1: train_score[SCORE_F1],
    SCORE_NUMBER_OF_QUESTIONS: train_score[SCORE_NUMBER_OF_QUESTIONS],
    'validation_' + SCORE_EXACT: validation_score[SCORE_EXACT],
    'validation_' + SCORE_F1: validation_score[SCORE_F1],
    'validation_' + SCORE_NUMBER_OF_QUESTIONS: validation_score[SCORE_NUMBER_OF_QUESTIONS],
    'random_' + SCORE_EXACT: random_validation_score[SCORE_EXACT],
    'random_' + SCORE_F1: random_validation_score[SCORE_F1],
    'random_' + SCORE_NUMBER_OF_QUESTIONS: random_validation_score[SCORE_NUMBER_OF_QUESTIONS]
})
