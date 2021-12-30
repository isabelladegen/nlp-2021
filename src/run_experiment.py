import wandb

from src.preprocessing_documents import load_documents_df, grounding_documents_for_dataframe
from src.preprocessing_rc import load_rc_dataset
from src.train_and_predict import BatchTrainer
from src.evaluate import *
from src.configurations import Configuration

run = wandb.init(project="test", notes="Checking of it's useful to log a per question score too",
                 tags=["another test run"], config=Configuration().as_dict())

# load document data for training
df = load_documents_df(wandb.config)

# pre-process grounding documents ready for training
grounding_documents = grounding_documents_for_dataframe(df)

# train a model for each grounding document
trainer = BatchTrainer(grounding_documents, wandb.config)

# get the current wand configuration in case there's a sweep running
config = Configuration(**wandb.config)

train_predictions = trainer.predict_answers_for(load_rc_dataset(config.predict_answers_rc_split, wandb.config))
train_score = train_predictions.squad2_score()

validation_predictions = trainer.predict_answers_for(
    load_rc_dataset(config.validate_answers_rc_split, wandb.config))
validation_score = validation_predictions.squad2_score()

random_validation_predictions = trainer.predict_random_spans_for(
    load_rc_dataset(config.random_answers_rc_split, wandb.config))
random_validation_score = random_validation_predictions.squad2_score()

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

train_scores_per_predictions = train_predictions.per_prediction_score()
train_table = wandb.Table(dataframe=train_scores_per_predictions)
run.log({"train_per_prediction_scores": train_table})

valid_scores_per_predictions = validation_predictions.per_prediction_score()
validation_table = wandb.Table(dataframe=valid_scores_per_predictions)
run.log({"validation_per_prediction_scores": validation_table})

random_scores_per_predictions = random_validation_predictions.per_prediction_score()
random_table = wandb.Table(dataframe=random_scores_per_predictions)
run.log({"random_per_prediction_scores": random_table})
