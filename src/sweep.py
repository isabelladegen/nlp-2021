import wandb
from tune_model_parameters_experiment import train
from src.evaluate import SCORE_F1
from src.configurations import QuestionPreProcessing

if __name__ == '__main__':
    parameters_to_try = {
        'pre_process_rc_question': {
            'values': [QuestionPreProcessing.default.value, QuestionPreProcessing.user_question_only.value]
        },
        'vector_size': {
            'values': [100, 150, 200]
        },
        'epochs': {
            'values': [150, 200, 250]
        },
        'dm': {
            'values': [0, 1]
        },
        'number_of_most_likely_docs': {
            'values': [1, 3]
        }
    }

    sweep_config_grid = {
        'name': 'test',
        'method': 'grid',
        'parameters': parameters_to_try
    }

    sweep_config_bayes = {
        'name': 'test',
        'method': 'bayes',
        'metric': {
            'goal': 'maximize',
            'name': SCORE_F1
        },
        'parameters': parameters_to_try,
        # 'early_terminate': {
        #     'type': 'hyperband',
        #     's': 2,
        #     'eta': 3,
        #     'max_iter': 27
        # }
    }

    sweep_id = wandb.sweep(sweep_config_grid, project="test")
    wandb.agent(sweep_id, function=train)
